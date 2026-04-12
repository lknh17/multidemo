"""
V18 - 商品属性提取 & 质量评估模型
===================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from config import ProductFullConfig, ProductAttributeConfig, QualityAssessConfig
from fine_grained import SimpleViTBackbone, FineGrainedConfig


class ProductAttributeModel(nn.Module):
    """商品属性多任务提取模型"""

    ATTRIBUTE_NAMES = ['category', 'brand', 'color', 'material', 'style']

    def __init__(self, config: ProductAttributeConfig):
        super().__init__()
        fg_cfg = FineGrainedConfig(
            d_model=config.d_model, n_heads=config.n_heads,
            image_size=config.image_size, patch_size=config.patch_size
        )
        self.encoder = SimpleViTBackbone(fg_cfg)

        # 多任务属性头
        self.category_head = nn.Linear(config.d_model, config.num_categories)
        self.brand_head = nn.Linear(config.d_model, config.num_brands)
        self.color_head = nn.Linear(config.d_model, config.num_colors)
        self.material_head = nn.Linear(config.d_model, config.num_materials)
        self.style_head = nn.Linear(config.d_model, config.num_styles)

        # 属性间关系建模
        self.attr_queries = nn.Parameter(torch.zeros(5, config.d_model))
        nn.init.trunc_normal_(self.attr_queries, std=0.02)
        self.attr_cross_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, batch_first=True
        )
        self.attr_norm = nn.LayerNorm(config.d_model)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(images)
        cls_feat = features[:, 0]      # [B, D]
        patch_feats = features[:, 1:]  # [B, N, D]

        # 属性间关系建模
        B = images.shape[0]
        queries = self.attr_queries.unsqueeze(0).expand(B, -1, -1)
        attr_feats, _ = self.attr_cross_attn(queries, patch_feats, patch_feats)
        attr_feats = self.attr_norm(attr_feats + queries)

        # 各属性预测
        return {
            'category': self.category_head(attr_feats[:, 0]),
            'brand': self.brand_head(attr_feats[:, 1]),
            'color': torch.sigmoid(self.color_head(attr_feats[:, 2])),
            'material': torch.sigmoid(self.material_head(attr_feats[:, 3])),
            'style': torch.sigmoid(self.style_head(attr_feats[:, 4])),
            'cls_feat': cls_feat,
        }


class QualityAssessmentModel(nn.Module):
    """商品图像质量评估模型"""

    QUALITY_DIMS = ['清晰度', '曝光', '构图', '美感', '合规']

    def __init__(self, config: QualityAssessConfig):
        super().__init__()
        fg_cfg = FineGrainedConfig(
            d_model=config.d_model, n_heads=config.n_heads,
            image_size=config.image_size, patch_size=config.patch_size
        )
        self.encoder = SimpleViTBackbone(fg_cfg)

        # 多维度质量评分头
        self.quality_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, config.num_quality_dims),
            nn.Sigmoid(),
        )

        # 整体质量分数
        self.overall_head = nn.Sequential(
            nn.Linear(config.d_model + config.num_quality_dims, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(images)
        cls_feat = features[:, 0]

        # 多维度评分
        dim_scores = self.quality_head(cls_feat)  # [B, 5]

        # 整体评分
        overall_input = torch.cat([cls_feat, dim_scores], dim=-1)
        overall_score = self.overall_head(overall_input)  # [B, 1]

        return {
            'dim_scores': dim_scores,
            'overall_score': overall_score.squeeze(-1),
        }

    def compute_rank_loss(self, scores_i: torch.Tensor, scores_j: torch.Tensor,
                          margin: float = 0.1) -> torch.Tensor:
        """排序学习损失：i 优于 j"""
        return F.relu(margin - (scores_i - scores_j)).mean()
