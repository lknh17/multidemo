"""
V18 - 细粒度视觉识别核心模块
=============================
1. AttentionCropper：基于注意力的判别区域裁剪
2. PartDetector：多零件检测器
3. MultiGranularityModel：多粒度特征学习
4. ArcFaceHead：ArcFace 度量学习
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from config import FineGrainedConfig


class SimpleViTBackbone(nn.Module):
    """ViT 风格骨干网络"""

    def __init__(self, config: FineGrainedConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(config.in_channels, config.d_model,
                                     config.patch_size, config.patch_size)
        n_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(config.d_model, config.n_heads,
                                       config.d_model * 4, batch_first=True)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.n_patches_side = config.image_size // config.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """获取最后一层注意力权重"""
        B = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, :tokens.shape[1]+1]

        for block in self.blocks:
            tokens = block(tokens)

        # 简化：用 CLS token 与 patches 的相似度作为注意力
        cls_feat = tokens[:, 0:1]
        patch_feats = tokens[:, 1:]
        attn = torch.matmul(cls_feat, patch_feats.transpose(-2, -1))
        attn = attn.squeeze(1)  # [B, N]
        attn = F.softmax(attn / math.sqrt(patch_feats.shape[-1]), dim=-1)
        H = W = self.n_patches_side
        return attn.reshape(B, H, W)


class AttentionCropper(nn.Module):
    """基于注意力权重裁剪判别性区域"""

    def __init__(self, target_size: int = 224, threshold: float = 0.5):
        super().__init__()
        self.target_size = target_size
        self.threshold = threshold

    def forward(self, attn_map: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_map: [B, H_p, W_p] patch 级注意力
            images: [B, C, H, W] 原始图像
        Returns:
            crops: [B, C, target_size, target_size]
        """
        B, H_p, W_p = attn_map.shape
        _, C, H, W = images.shape
        scale_h, scale_w = H / H_p, W / W_p

        crops = []
        for b in range(B):
            am = attn_map[b]
            thresh = am.max() * self.threshold
            mask = am > thresh

            # 找 bounding box
            rows = mask.any(dim=1)
            cols = mask.any(dim=0)

            if rows.any() and cols.any():
                y0 = int(rows.nonzero(as_tuple=True)[0].min().item() * scale_h)
                y1 = int((rows.nonzero(as_tuple=True)[0].max().item() + 1) * scale_h)
                x0 = int(cols.nonzero(as_tuple=True)[0].min().item() * scale_w)
                x1 = int((cols.nonzero(as_tuple=True)[0].max().item() + 1) * scale_w)
                crop = images[b:b+1, :, y0:y1, x0:x1]
            else:
                crop = images[b:b+1]

            crop = F.interpolate(crop, size=(self.target_size, self.target_size),
                               mode='bilinear', align_corners=False)
            crops.append(crop)

        return torch.cat(crops, dim=0)


class PartDetector(nn.Module):
    """多零件检测器：用可学习 Query 检测判别性局部区域"""

    def __init__(self, d_model: int, num_parts: int):
        super().__init__()
        self.num_parts = num_parts
        self.part_queries = nn.Parameter(torch.zeros(num_parts, d_model))
        nn.init.trunc_normal_(self.part_queries, std=0.02)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_features: [B, N, D]
        Returns:
            part_feats: [B, K, D]
            attn_maps: [B, K, N]
        """
        B = patch_features.shape[0]
        queries = self.part_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention
        attn = torch.matmul(queries, patch_features.transpose(-2, -1))
        attn = attn / math.sqrt(patch_features.shape[-1])
        attn_maps = F.softmax(attn, dim=-1)  # [B, K, N]

        # 加权聚合
        part_feats = torch.matmul(attn_maps, patch_features)  # [B, K, D]

        return part_feats, attn_maps


class MultiGranularityModel(nn.Module):
    """多粒度细粒度识别模型"""

    def __init__(self, config: FineGrainedConfig):
        super().__init__()
        self.config = config
        self.backbone = SimpleViTBackbone(config)
        self.part_detector = PartDetector(config.d_model, config.num_parts)
        self.cropper = AttentionCropper()

        # 分类头
        self.global_head = nn.Linear(config.d_model, config.num_classes)
        self.part_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.num_classes)
            for _ in range(config.num_parts)
        ])
        self.fused_head = nn.Linear(config.d_model * (1 + config.num_parts), config.num_classes)

        # 多样性正则
        self.diversity_weight = 0.1

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        global_feat = features[:, 0]          # [B, D]
        patch_feats = features[:, 1:]         # [B, N, D]

        # 零件检测
        part_feats, attn_maps = self.part_detector(patch_feats)

        # 全局分类
        global_logits = self.global_head(global_feat)

        # 局部分类
        part_logits_list = []
        for k in range(self.config.num_parts):
            part_logits_list.append(self.part_heads[k](part_feats[:, k]))

        # 融合分类
        fused = torch.cat([global_feat, part_feats.flatten(1)], dim=-1)
        fused_logits = self.fused_head(fused)

        # 多样性损失
        diversity_loss = self._diversity_loss(attn_maps)

        return {
            'global_logits': global_logits,
            'part_logits': part_logits_list,
            'fused_logits': fused_logits,
            'attn_maps': attn_maps,
            'diversity_loss': diversity_loss,
        }

    def _diversity_loss(self, attn_maps: torch.Tensor) -> torch.Tensor:
        # attn_maps: [B, K, N]
        # 鼓励不同 part 关注不同区域
        A = attn_maps  # [B, K, N]
        gram = torch.matmul(A, A.transpose(-2, -1))  # [B, K, K]
        I = torch.eye(A.shape[1], device=A.device).unsqueeze(0)
        return ((gram - I) ** 2).mean()


class ArcFaceHead(nn.Module):
    """ArcFace 度量学习头"""

    def __init__(self, d_model: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        weight = F.normalize(self.weight, dim=-1)

        cosine = F.linear(features, weight)  # [B, C]
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))

        # 只对正确类别加 margin
        one_hot = F.one_hot(labels, self.num_classes).float()
        target_theta = theta * one_hot + self.margin * one_hot
        target_cosine = torch.cos(target_theta)

        logits = cosine * (1 - one_hot) + target_cosine * one_hot
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)
