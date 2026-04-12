"""
V24 - 内容安全核心模块
======================
1. ContentClassifier：8 类内容安全分类
2. ToxicityScorer：文本 + 图像毒性评分
3. WatermarkEmbedder：水印嵌入与检测
4. AdversarialAttacker：FGSM / PGD 攻击
5. InputSanitizer：输入清洗
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from config import (SafetyClassifierConfig, ToxicityConfig,
                    WatermarkConfig, AdversarialConfig)


# ============================================================
# 简易 ViT 骨干（复用模式）
# ============================================================
class SimpleViTBackbone(nn.Module):
    """轻量 ViT 骨干"""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 image_size: int = 224, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, d_model, patch_size, patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================
# 简易文本编码器
# ============================================================
class SimpleTextEncoder(nn.Module):
    """轻量文本编码器"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================
# 1. ContentClassifier — 8 类内容安全分类
# ============================================================
class ContentClassifier(nn.Module):
    """多标签内容安全分类器（NSFW/暴力/垃圾/仇恨/自残/恐怖/违禁/正常）"""

    CATEGORY_NAMES = ['nsfw', 'violence', 'spam', 'hate',
                      'self_harm', 'terrorism', 'contraband', 'normal']

    def __init__(self, config: SafetyClassifierConfig):
        super().__init__()
        self.config = config
        self.backbone = SimpleViTBackbone(
            config.d_model, config.n_heads, config.n_layers,
            config.image_size, config.patch_size, config.in_channels
        )
        # 多标签分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, config.num_categories),
        )
        # 每类独立阈值
        self.thresholds = nn.Parameter(
            torch.full((config.num_categories,), config.threshold),
            requires_grad=False
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        cls_feat = features[:, 0]  # [B, D]
        logits = self.classifier(cls_feat)  # [B, K]
        probs = torch.sigmoid(logits)
        return {
            'logits': logits,
            'probs': probs,
            'cls_feat': cls_feat,
            'predictions': (probs > self.thresholds.unsqueeze(0)).float(),
        }

    def update_thresholds(self, new_thresholds: torch.Tensor):
        """用 F1 最优阈值更新"""
        self.thresholds.data.copy_(new_thresholds)


# ============================================================
# 2. ToxicityScorer — 多模态毒性检测
# ============================================================
class ToxicityScorer(nn.Module):
    """文本 + 图像融合毒性评分"""

    TOXICITY_DIMS = ['insult', 'threat', 'obscene', 'discrimination', 'hate', 'normal']

    def __init__(self, config: ToxicityConfig):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(
            config.vocab_size, config.text_model_dim,
            n_heads=4, n_layers=2, max_seq_len=config.max_seq_len
        )
        self.image_encoder = SimpleViTBackbone(
            config.image_model_dim, config.n_heads, 2,
            config.image_size, config.patch_size, config.in_channels
        )
        # 跨模态融合
        self.cross_attn = nn.MultiheadAttention(
            config.fused_dim, config.n_heads, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(config.fused_dim)
        # 毒性评分头
        self.toxicity_head = nn.Sequential(
            nn.Linear(config.fused_dim, config.fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.fused_dim // 2, config.num_toxicity_dims),
            nn.Sigmoid(),
        )
        # 总体毒性分
        self.overall_head = nn.Sequential(
            nn.Linear(config.fused_dim + config.num_toxicity_dims, config.fused_dim // 2),
            nn.GELU(),
            nn.Linear(config.fused_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor,
                images: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_feats = self.text_encoder(token_ids)   # [B, L, D]
        image_feats = self.image_encoder(images)    # [B, 1+N, D]

        # 文本 query 图像
        fused, _ = self.cross_attn(text_feats, image_feats, image_feats)
        fused = self.fusion_norm(fused + text_feats)
        pooled = fused.mean(dim=1)  # [B, D]

        dim_scores = self.toxicity_head(pooled)  # [B, 6]
        overall_input = torch.cat([pooled, dim_scores], dim=-1)
        overall = self.overall_head(overall_input).squeeze(-1)  # [B]

        return {
            'dim_scores': dim_scores,
            'overall_score': overall,
            'pooled_feat': pooled,
        }


# ============================================================
# 3. WatermarkEmbedder — 水印嵌入与检测
# ============================================================
class WatermarkEmbedder(nn.Module):
    """神经网络水印嵌入与检测（模拟 DWT 域嵌入）"""

    def __init__(self, config: WatermarkConfig):
        super().__init__()
        self.config = config
        n_patches = (config.image_size // config.patch_size) ** 2

        # 编码器：将水印比特编码为嵌入空间
        self.watermark_encoder = nn.Sequential(
            nn.Linear(config.watermark_bits, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        # 嵌入器：将水印融入图像特征
        self.embed_net = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, config.in_channels, 3, padding=1),
            nn.Tanh(),
        )
        # 检测器：从图像中提取水印
        self.detector = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.watermark_bits),
        )

    def embed(self, images: torch.Tensor,
              watermark: torch.Tensor) -> torch.Tensor:
        """嵌入水印到图像中"""
        # watermark: [B, watermark_bits] ∈ {-1, +1}
        residual = self.embed_net(images)  # [B, C, H, W]
        watermarked = images + self.config.alpha * residual
        return watermarked.clamp(0, 1)

    def detect(self, images: torch.Tensor) -> torch.Tensor:
        """从图像中检测水印"""
        logits = self.detector(images)  # [B, watermark_bits]
        return logits

    def forward(self, images: torch.Tensor,
                watermark: torch.Tensor) -> Dict[str, torch.Tensor]:
        watermarked = self.embed(images, watermark)
        detected = self.detect(watermarked)
        # 计算比特准确率
        predicted_bits = (detected > 0).float()
        original_bits = (watermark > 0).float()
        bit_acc = (predicted_bits == original_bits).float().mean()
        # 图像质量损失（PSNR 相关）
        mse = F.mse_loss(watermarked, images)
        return {
            'watermarked': watermarked,
            'detected_logits': detected,
            'bit_accuracy': bit_acc,
            'embed_mse': mse,
        }


# ============================================================
# 4. AdversarialAttacker — FGSM / PGD 攻击
# ============================================================
class AdversarialAttacker:
    """对抗攻击器（FGSM 和 PGD）"""

    def __init__(self, config: AdversarialConfig):
        self.epsilon = config.epsilon
        self.attack_steps = config.attack_steps
        self.step_size = config.step_size

    def fgsm(self, model: nn.Module, images: torch.Tensor,
             labels: torch.Tensor, loss_fn=None) -> torch.Tensor:
        """FGSM 单步攻击"""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        images_adv = images.clone().detach().requires_grad_(True)
        outputs = model(images_adv)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss = loss_fn(logits, labels)
        loss.backward()
        perturbation = self.epsilon * images_adv.grad.sign()
        adv = (images + perturbation).clamp(0, 1)
        return adv.detach()

    def pgd(self, model: nn.Module, images: torch.Tensor,
            labels: torch.Tensor, loss_fn=None) -> torch.Tensor:
        """PGD 多步迭代攻击"""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        adv = images.clone().detach()
        adv = adv + torch.empty_like(adv).uniform_(-self.epsilon, self.epsilon)
        adv = adv.clamp(0, 1)

        for _ in range(self.attack_steps):
            adv.requires_grad_(True)
            outputs = model(adv)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = loss_fn(logits, labels)
            loss.backward()
            grad_sign = adv.grad.sign()
            adv = adv.detach() + self.step_size * grad_sign
            # 投影回 ε-ball
            delta = (adv - images).clamp(-self.epsilon, self.epsilon)
            adv = (images + delta).clamp(0, 1)

        return adv.detach()


# ============================================================
# 5. InputSanitizer — 输入清洗
# ============================================================
class InputSanitizer(nn.Module):
    """输入清洗器：检测并移除潜在对抗扰动"""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        # 高斯平滑
        self.smoother = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size, padding=kernel_size // 2, groups=3, bias=False),
            nn.BatchNorm2d(3),
        )
        # 初始化为高斯核
        with torch.no_grad():
            k = kernel_size
            gauss = torch.zeros(k, k)
            center = k // 2
            for i in range(k):
                for j in range(k):
                    gauss[i, j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / 2.0)
            gauss = gauss / gauss.sum()
            self.smoother[0].weight.data = gauss.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """对输入做轻度平滑以消除高频对抗扰动"""
        smoothed = self.smoother(images)
        return smoothed.clamp(0, 1)
