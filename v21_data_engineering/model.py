"""
V21 - 多模态数据工程 - 模型模块
================================
实现增强训练器（CutMix/MixUp/RandAugment 包装器）、
课程学习调度器、合成数据生成器。
"""

import math
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FullConfig


# ============================================================
# 1. 简单 ViT 分类器（作为训练/增强的基础模型）
# ============================================================
class SimplePatchClassifier(nn.Module):
    """
    简单的 Patch-based 图像分类器。
    
    结构：图像 → 分 patch → 线性投影 → Transformer → 分类头
    作为数据工程实验的基础模型。
    """
    
    def __init__(self, config: FullConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = config.in_channels * config.patch_size ** 2
        
        self.patch_embed = nn.Linear(patch_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.d_model) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 图像
        Returns:
            logits: [B, num_classes]
        """
        B = x.size(0)
        ps = self.config.patch_size
        
        # 分 patch: [B, C, H, W] → [B, num_patches, patch_dim]
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(B, -1, self.config.in_channels * ps * ps)
        
        # 线性投影
        tokens = self.patch_embed(patches)  # [B, N, d_model]
        
        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        
        # Transformer 编码
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        
        # CLS token 分类
        logits = self.head(tokens[:, 0])
        return logits


# ============================================================
# 2. MixUp 增强
#    全局像素线性插值 + 标签混合
# ============================================================
class MixUp:
    """
    MixUp 数据增强。
    
    公式：
        x̃ = λ·x_i + (1-λ)·x_j
        ỹ = λ·y_i + (1-λ)·y_j
    其中 λ ~ Beta(α, α)
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一个 batch 应用 MixUp。
        
        Args:
            images: [B, C, H, W]
            labels: [B] (整数标签)
            num_classes: 类别数
        
        Returns:
            (mixed_images, mixed_labels)
            mixed_labels 是 one-hot 形式: [B, num_classes]
        """
        B = images.size(0)
        
        # 从 Beta(α, α) 采样混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机打乱索引作为混合对
        perm = torch.randperm(B, device=images.device)
        
        # 图像混合
        mixed_images = lam * images + (1 - lam) * images[perm]
        
        # 标签转 one-hot 后混合
        labels_one_hot = F.one_hot(labels, num_classes).float()
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[perm]
        
        return mixed_images, mixed_labels


# ============================================================
# 3. CutMix 增强
#    矩形区域替换 + 按面积比混合标签
# ============================================================
class CutMix:
    """
    CutMix 数据增强。
    
    公式：
        x̃ = M ⊙ x_i + (1-M) ⊙ x_j
        ỹ = λ·y_i + (1-λ)·y_j
    其中 M 是二值掩码，λ 为保留区域面积比。
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def _rand_bbox(
        self, H: int, W: int, lam: float
    ) -> Tuple[int, int, int, int]:
        """
        随机生成矩形区域。
        
        矩形面积比 = 1 - λ
        r_w = W * sqrt(1-λ), r_h = H * sqrt(1-λ)
        """
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # 随机中心点
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        
        # 裁剪到图像范围内
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        
        return x1, y1, x2, y2
    
    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一个 batch 应用 CutMix。
        
        Args:
            images: [B, C, H, W]
            labels: [B]
            num_classes: 类别数
        
        Returns:
            (mixed_images, mixed_labels)
        """
        B, C, H, W = images.shape
        
        if random.random() > self.prob:
            # 不应用 CutMix
            return images, F.one_hot(labels, num_classes).float()
        
        lam = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(B, device=images.device)
        
        x1, y1, x2, y2 = self._rand_bbox(H, W, lam)
        
        # 替换矩形区域
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        
        # 根据实际替换面积重新计算 λ
        lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
        
        # 标签混合
        labels_one_hot = F.one_hot(labels, num_classes).float()
        mixed_labels = lam_actual * labels_one_hot + (1 - lam_actual) * labels_one_hot[perm]
        
        return mixed, mixed_labels


# ============================================================
# 4. RandAugment 增强
#    随机选择 N 个变换，每个强度为 M
# ============================================================
class RandAugment:
    """
    RandAugment: 从变换池中随机选择 N 个变换，统一强度 M。
    
    优点：只有 2 个超参数（N 和 M），搜索空间极小。
    这里用纯 tensor 操作实现简化版本。
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n  # 每次应用的变换数
        self.m = m  # 变换强度 (0-30)
        self.magnitude = m / 30.0  # 归一化到 [0, 1]
    
    def _brightness(self, img: torch.Tensor) -> torch.Tensor:
        """亮度调整"""
        factor = 1.0 + self.magnitude * (random.random() * 2 - 1)
        return torch.clamp(img * factor, 0, 1)
    
    def _contrast(self, img: torch.Tensor) -> torch.Tensor:
        """对比度调整"""
        mean = img.mean(dim=(-2, -1), keepdim=True)
        factor = 1.0 + self.magnitude * (random.random() * 2 - 1)
        return torch.clamp(mean + (img - mean) * factor, 0, 1)
    
    def _add_noise(self, img: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(img) * self.magnitude * 0.1
        return torch.clamp(img + noise, 0, 1)
    
    def _horizontal_flip(self, img: torch.Tensor) -> torch.Tensor:
        """水平翻转"""
        return img.flip(-1)
    
    def _vertical_flip(self, img: torch.Tensor) -> torch.Tensor:
        """垂直翻转"""
        return img.flip(-2)
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        对一个 batch 应用 RandAugment。
        
        Args:
            images: [B, C, H, W]，值范围 [0, 1]
        
        Returns:
            augmented images: [B, C, H, W]
        """
        transforms = [
            self._brightness,
            self._contrast,
            self._add_noise,
            self._horizontal_flip,
            self._vertical_flip,
        ]
        
        augmented = images.clone()
        # 随机选择 N 个变换
        selected = random.sample(transforms, min(self.n, len(transforms)))
        
        for transform in selected:
            augmented = transform(augmented)
        
        return augmented


# ============================================================
# 5. 增强训练器
#    将 CutMix / MixUp / RandAugment 整合到训练流程中
# ============================================================
class AugmentedTrainer:
    """
    增强训练器：在训练循环中自动应用数据增强。
    
    支持的增强策略：
    - MixUp: 像素级线性混合
    - CutMix: 矩形区域替换
    - RandAugment: 随机变换组合
    """
    
    def __init__(self, config: FullConfig):
        self.config = config
        self.mixup = MixUp(alpha=config.augmentation.mixup_alpha)
        self.cutmix = CutMix(
            alpha=config.augmentation.cutmix_alpha,
            prob=config.augmentation.cutmix_prob,
        )
        self.randaug = RandAugment(
            n=config.augmentation.randaug_n,
            m=config.augmentation.randaug_m,
        )
    
    def apply_augmentation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        aug_type: str = "mixup",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用指定的增强策略。
        
        Args:
            images: [B, C, H, W]
            labels: [B]
            aug_type: "mixup" / "cutmix" / "randaug" / "mixed"
        
        Returns:
            (augmented_images, mixed_labels)
            mixed_labels: [B, num_classes] (one-hot 或软标签)
        """
        nc = self.config.num_classes
        
        if aug_type == "mixup":
            return self.mixup(images, labels, nc)
        elif aug_type == "cutmix":
            return self.cutmix(images, labels, nc)
        elif aug_type == "randaug":
            aug_images = self.randaug(images)
            return aug_images, F.one_hot(labels, nc).float()
        elif aug_type == "mixed":
            # 随机选择一种增强
            choice = random.choice(["mixup", "cutmix", "randaug"])
            return self.apply_augmentation(images, labels, choice)
        else:
            return images, F.one_hot(labels, nc).float()


# ============================================================
# 6. 课程学习调度器
#    根据难度和训练进度动态选择训练数据
# ============================================================
class CurriculumScheduler:
    """
    课程学习调度器。
    
    核心思想：先学简单样本，再逐步引入困难样本。
    
    组件：
    1. 难度评分函数：衡量每个样本的难度
    2. 节奏函数 λ(t)：控制当前使用数据的比例
    3. 数据选择：按难度排序，只使用前 λ(t) 比例的数据
    """
    
    def __init__(self, config: FullConfig, total_epochs: int):
        self.config = config
        self.curriculum = config.curriculum
        self.total_epochs = total_epochs
        self.sample_difficulties: Optional[torch.Tensor] = None
    
    def compute_difficulty(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算样本难度。
        
        根据 difficulty_metric 配置选择不同的衡量方式：
        - loss: 单样本 loss（越高越难）
        - confidence: 1 - 预测概率（越低越难）
        - entropy: 预测分布的熵（越高越难）
        """
        model.eval()
        with torch.no_grad():
            logits = model(images.to(device))
            probs = F.softmax(logits, dim=-1)
            
            if self.curriculum.difficulty_metric == "loss":
                losses = F.cross_entropy(logits, labels.to(device), reduction='none')
                difficulties = losses.cpu()
            
            elif self.curriculum.difficulty_metric == "confidence":
                confidence = probs.gather(1, labels.to(device).unsqueeze(1)).squeeze(1)
                difficulties = (1.0 - confidence).cpu()
            
            elif self.curriculum.difficulty_metric == "entropy":
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
                difficulties = entropy.cpu()
            
            else:
                difficulties = torch.zeros(len(labels))
        
        model.train()
        return difficulties
    
    def pacing_function(self, epoch: int) -> float:
        """
        节奏函数：根据当前 epoch 计算应使用的数据比例 λ(t)。
        
        t = epoch / total_epochs (训练进度)
        λ_0 = initial_fraction (初始数据比例)
        T_w = warmup_epochs / total_epochs (warmup 比例)
        
        linear:  λ(t) = λ_0 + (1-λ_0) · min(1, t/T_w)
        root:    λ(t) = λ_0 + (1-λ_0) · min(1, sqrt(t/T_w))
        step:    λ(t) = λ_0 + (1-λ_0) · min(1, floor(t·K/T_w) / K)
        """
        t = epoch / max(self.total_epochs, 1)
        lam0 = self.curriculum.initial_fraction
        tw = self.curriculum.warmup_epochs / max(self.total_epochs, 1)
        
        if tw <= 0:
            return 1.0
        
        ratio = min(1.0, t / tw)
        
        if self.curriculum.pacing_function == "linear":
            return lam0 + (1.0 - lam0) * ratio
        
        elif self.curriculum.pacing_function == "root":
            return lam0 + (1.0 - lam0) * math.sqrt(ratio)
        
        elif self.curriculum.pacing_function == "step":
            K = self.curriculum.difficulty_bins
            return lam0 + (1.0 - lam0) * min(1.0, math.floor(ratio * K) / K)
        
        else:
            return 1.0
    
    def select_samples(
        self,
        difficulties: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        根据难度和节奏函数选择训练样本。
        
        Args:
            difficulties: [N] 每个样本的难度分数
            epoch: 当前 epoch
        
        Returns:
            selected_indices: 选中的样本索引
        """
        fraction = self.pacing_function(epoch)
        n_select = max(1, int(len(difficulties) * fraction))
        
        # 按难度排序（从易到难）
        sorted_indices = torch.argsort(difficulties)
        
        # 选择前 fraction 比例的样本（最简单的）
        selected = sorted_indices[:n_select]
        
        return selected


# ============================================================
# 7. 合成数据生成器
#    简单的条件生成（基于噪声 + 类别条件）
# ============================================================
class SyntheticGenerator(nn.Module):
    """
    简单的条件合成数据生成器。
    
    架构：噪声向量 + 类别嵌入 → MLP → 合成图像
    
    这是一个演示用的简化版本，真实场景中会使用 GAN / Diffusion 等更强的生成模型。
    """
    
    def __init__(self, config: FullConfig, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        
        img_dim = config.in_channels * config.image_size * config.image_size
        
        # 类别嵌入
        self.class_embed = nn.Embedding(config.num_classes, latent_dim)
        
        # 生成器 MLP
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, img_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )
    
    def forward(
        self, num_samples: int, class_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成合成数据。
        
        Args:
            num_samples: 生成样本数
            class_labels: [N] 指定类别，如果为 None 则随机
        
        Returns:
            (images, labels)
            images: [N, C, H, W]
            labels: [N]
        """
        device = next(self.parameters()).device
        
        if class_labels is None:
            class_labels = torch.randint(
                0, self.config.num_classes, (num_samples,), device=device
            )
        else:
            class_labels = class_labels.to(device)
        
        # 随机噪声
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # 类别条件
        c = self.class_embed(class_labels)
        
        # 拼接噪声和条件
        z_cond = torch.cat([z, c], dim=-1)
        
        # 生成
        flat_images = self.generator(z_cond)
        images = flat_images.view(
            num_samples,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        
        return images, class_labels


if __name__ == "__main__":
    config = FullConfig()
    
    # ---- 测试基础模型 ----
    print("=" * 50)
    print("SimplePatchClassifier 测试")
    print("=" * 50)
    
    model = SimplePatchClassifier(config)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    print(f"输入: {x.shape} → 输出: {logits.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    
    # ---- 测试 MixUp ----
    print("\n" + "=" * 50)
    print("MixUp 测试")
    print("=" * 50)
    
    mixup = MixUp(alpha=0.2)
    labels = torch.randint(0, 10, (4,))
    mixed_img, mixed_lbl = mixup(x, labels, 10)
    print(f"Mixed images: {mixed_img.shape}")
    print(f"Mixed labels: {mixed_lbl.shape}, sum={mixed_lbl.sum(dim=-1)}")
    
    # ---- 测试 CutMix ----
    print("\n" + "=" * 50)
    print("CutMix 测试")
    print("=" * 50)
    
    cutmix = CutMix(alpha=1.0, prob=1.0)
    cut_img, cut_lbl = cutmix(x, labels, 10)
    print(f"CutMix images: {cut_img.shape}")
    print(f"CutMix labels: {cut_lbl.shape}")
    
    # ---- 测试 RandAugment ----
    print("\n" + "=" * 50)
    print("RandAugment 测试")
    print("=" * 50)
    
    randaug = RandAugment(n=2, m=9)
    aug_img = randaug(torch.rand(4, 3, 32, 32))
    print(f"Augmented: {aug_img.shape}, range=[{aug_img.min():.3f}, {aug_img.max():.3f}]")
    
    # ---- 测试合成生成 ----
    print("\n" + "=" * 50)
    print("SyntheticGenerator 测试")
    print("=" * 50)
    
    gen = SyntheticGenerator(config)
    syn_img, syn_lbl = gen(8)
    print(f"合成图像: {syn_img.shape}, 标签: {syn_lbl.tolist()}")
