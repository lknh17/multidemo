"""
V20 - 知识增强多模态嵌入模型
=============================
核心模块：
1. KGEnhancedVisualModel：ViT + KG 注意力
2. KGAugmentedRetrieval：KG 增强的跨模态检索
3. KnowledgeDistillModel：从 KG 到视觉模型的知识蒸馏

参考：
- ERNIE-ViL (AAAI 2021)
- K-LITE (NeurIPS 2022)
- KEPLER (TACL 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from config import (
    KnowledgeEmbeddingFullConfig,
    KGEnhancedEmbeddingConfig,
    KnowledgeGraphConfig,
)
from kg_modules import TransEEmbedding, KGAttentionLayer


# ============================================================
#  ViT Patch 嵌入
# ============================================================

class PatchEmbedding(nn.Module):
    """ViT 风格 Patch 嵌入"""

    def __init__(self, config: KGEnhancedEmbeddingConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.in_channels, config.visual_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        n_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.visual_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, config.visual_dim) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            tokens: [B, N+1, D] (含 CLS token)
        """
        B = images.shape[0]
        patches = self.patch_embed(images)  # [B, D, H', W']
        patches = patches.flatten(2).permute(0, 2, 1)  # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # [B, N+1, D]
        tokens = tokens + self.pos_embed

        return tokens


# ============================================================
#  KG 增强 Transformer Block
# ============================================================

class KGEnhancedBlock(nn.Module):
    """KG 增强的 Transformer Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.kg_attn = KGAttentionLayer(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                kg_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.kg_attn(self.norm1(x), kg_embeds)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
#  KG 增强视觉模型
# ============================================================

class KGEnhancedVisualModel(nn.Module):
    """
    ViT + KG 注意力
    
    核心：在 Transformer 的每一层都可以关注 KG 实体嵌入
    视觉 token 通过注意力机制选择性地吸收实体知识
    """

    def __init__(self, config: KnowledgeEmbeddingFullConfig):
        super().__init__()
        self.config = config
        ec = config.kg_embed

        # ViT Patch 嵌入
        self.patch_embed = PatchEmbedding(ec)

        # KG 嵌入（预训练或联合训练）
        self.kg_embed = TransEEmbedding(config.kg)

        # KG → 视觉空间投影
        self.kg_proj = nn.Linear(config.kg.d_model, ec.visual_dim)

        # KG 增强 Transformer
        self.blocks = nn.ModuleList([
            KGEnhancedBlock(ec.visual_dim, ec.n_heads, ec.d_ff, ec.dropout)
            for _ in range(ec.n_layers)
        ])

        self.norm = nn.LayerNorm(ec.visual_dim)

        # 分类头
        self.cls_head = nn.Linear(ec.visual_dim, config.kg.num_relations)

    def forward(self, images: torch.Tensor,
                entity_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W]
            entity_ids: [B, E] 关联实体 ID
        """
        # 视觉编码
        x = self.patch_embed(images)  # [B, N+1, D]

        # KG 实体嵌入
        kg_embeds = None
        if entity_ids is not None:
            raw_kg = self.kg_embed.get_entity_embeddings(entity_ids)  # [B, E, D_kg]
            kg_embeds = self.kg_proj(raw_kg)  # [B, E, D_v]

        # Transformer with KG attention
        for block in self.blocks:
            x = block(x, kg_embeds)

        x = self.norm(x)
        cls_output = x[:, 0]  # CLS token

        return {
            'cls_output': cls_output,          # [B, D]
            'features': x,                     # [B, N+1, D]
            'logits': self.cls_head(cls_output),  # [B, num_relations]
        }


# ============================================================
#  KG 增强检索模型
# ============================================================

class KGAugmentedRetrieval(nn.Module):
    """
    KG 增强的跨模态检索
    
    视觉嵌入 + KG 嵌入 → 融合嵌入 → 对比学习检索
    """

    def __init__(self, config: KnowledgeEmbeddingFullConfig):
        super().__init__()
        self.config = config
        ec = config.kg_embed

        # 视觉编码器
        self.visual_encoder = KGEnhancedVisualModel(config)

        # 文本编码器（简化）
        self.text_embed = nn.Embedding(5000, ec.text_dim)
        self.text_pos = nn.Embedding(128, ec.text_dim)
        text_layer = nn.TransformerEncoderLayer(
            ec.text_dim, ec.n_heads, ec.d_ff, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=2)

        # 投影到共享空间
        self.visual_proj = nn.Sequential(
            nn.Linear(ec.visual_dim, ec.fusion_dim),
            nn.ReLU(),
            nn.Linear(ec.fusion_dim, ec.fusion_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(ec.text_dim, ec.fusion_dim),
            nn.ReLU(),
            nn.Linear(ec.fusion_dim, ec.fusion_dim),
        )

        # KG 增强投影
        self.kg_boost_proj = nn.Sequential(
            nn.Linear(config.kg.d_model, ec.fusion_dim),
            nn.ReLU(),
            nn.Linear(ec.fusion_dim, ec.fusion_dim),
        )

        # 温度参数
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def encode_visual(self, images: torch.Tensor,
                     entity_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码视觉 + KG → 融合嵌入"""
        vis_out = self.visual_encoder(images, entity_ids)
        vis_feat = self.visual_proj(vis_out['cls_output'])  # [B, D]

        if entity_ids is not None:
            kg_raw = self.visual_encoder.kg_embed.get_entity_embeddings(entity_ids)
            kg_feat = self.kg_boost_proj(kg_raw.mean(dim=1))  # [B, D]
            vis_feat = vis_feat + 0.3 * kg_feat  # KG boost

        return F.normalize(vis_feat, p=2, dim=-1)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.text_embed(token_ids) + self.text_pos(pos)
        x = self.text_encoder(x)
        text_feat = self.text_proj(x[:, 0])  # CLS
        return F.normalize(text_feat, p=2, dim=-1)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor,
                entity_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        vis_embed = self.encode_visual(images, entity_ids)  # [B, D]
        txt_embed = self.encode_text(token_ids)              # [B, D]

        # 对比学习
        logit_scale = self.logit_scale.exp().clamp(max=100)
        sim_v2t = logit_scale * vis_embed @ txt_embed.T  # [B, B]
        sim_t2v = sim_v2t.T

        return {
            'vis_embed': vis_embed,
            'txt_embed': txt_embed,
            'sim_v2t': sim_v2t,
            'sim_t2v': sim_t2v,
        }


# ============================================================
#  知识蒸馏模型
# ============================================================

class SimpleVisualModel(nn.Module):
    """纯视觉学生模型（无 KG）"""

    def __init__(self, config: KGEnhancedEmbeddingConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)

        block = nn.TransformerEncoderLayer(
            config.visual_dim, config.n_heads, config.d_ff,
            dropout=config.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.visual_dim)
        self.head = nn.Linear(config.visual_dim, 200)  # num_classes

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(images)
        x = self.encoder(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        return {
            'cls_output': cls_output,
            'features': x,
            'logits': self.head(cls_output),
        }


class KnowledgeDistillModel(nn.Module):
    """
    知识蒸馏：从 KG-enhanced 教师到纯视觉学生
    
    教师：KGEnhancedVisualModel（冻结）
    学生：SimpleVisualModel（训练）
    
    损失：task_loss + α * distill_loss + β * feat_loss
    """

    def __init__(self, config: KnowledgeEmbeddingFullConfig):
        super().__init__()
        self.config = config

        # 教师（冻结）
        self.teacher = KGEnhancedVisualModel(config)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 学生
        self.student = SimpleVisualModel(config.kg_embed)

        # 特征对齐投影
        self.align_proj = nn.Linear(config.kg_embed.visual_dim, config.kg_embed.visual_dim)

        self.temperature = config.distill_temperature
        self.alpha = config.distill_alpha

    def compute_distill_loss(self, student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor) -> torch.Tensor:
        """软标签 KL 散度"""
        T = self.temperature
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        return T * T * F.kl_div(student_probs, teacher_probs, reduction='batchmean')

    def compute_feature_loss(self, student_feat: torch.Tensor,
                             teacher_feat: torch.Tensor) -> torch.Tensor:
        """特征对齐损失"""
        projected = self.align_proj(teacher_feat)
        return F.mse_loss(student_feat, projected.detach())

    def forward(self, images: torch.Tensor,
                entity_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 教师推理（不计算梯度）
        with torch.no_grad():
            teacher_out = self.teacher(images, entity_ids)

        # 学生推理
        student_out = self.student(images)

        outputs = {
            'student_logits': student_out['logits'],
            'teacher_logits': teacher_out['logits'],
            'student_features': student_out['cls_output'],
            'teacher_features': teacher_out['cls_output'],
        }

        # 计算损失
        if labels is not None:
            task_loss = F.cross_entropy(student_out['logits'], labels)
            distill_loss = self.compute_distill_loss(
                student_out['logits'], teacher_out['logits']
            )
            feat_loss = self.compute_feature_loss(
                student_out['cls_output'], teacher_out['cls_output']
            )

            total_loss = ((1 - self.alpha) * task_loss +
                         self.alpha * distill_loss +
                         0.1 * feat_loss)

            outputs.update({
                'loss': total_loss,
                'task_loss': task_loss,
                'distill_loss': distill_loss,
                'feat_loss': feat_loss,
            })

        return outputs
