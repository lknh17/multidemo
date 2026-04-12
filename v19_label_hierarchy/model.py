"""
V19 - 层级标签理解模型
======================
核心模块：
1. HierarchicalClassifier：coarse→mid→fine 三级级联分类器
2. ConstrainedMultiLabelModel：层级约束多标签分类模型
3. LabelEmbeddingModel：视觉-标签联合嵌入模型

参考：
- Hierarchical Multi-label Classification (Silla & Freitas, 2011)
- Poincaré Embeddings for Learning Hierarchical Representations (Nickel & Kiela, 2017)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from config import LabelHierarchyFullConfig, LabelHierarchyConfig, LabelEmbeddingConfig, MultiLabelConfig
from taxonomy import (
    TaxonomyTree, HierarchicalSoftmax, LabelPropagationGNN,
    HyperbolicEmbedding, poincare_distance, exp_map_zero, project_to_ball,
    LabelConsistencyChecker,
)


# ============================================================
#  简易视觉编码器（ViT 风格）
# ============================================================

class SimplePatchEncoder(nn.Module):
    """简易 Patch 编码器：图像 → 特征向量"""

    def __init__(self, config: LabelHierarchyConfig):
        super().__init__()
        n_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            config.in_channels, config.d_model,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, config.d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.d_ff,
            dropout=config.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            features: [B, D] 全局特征（CLS token）
        """
        B = images.shape[0]
        # Patch embedding
        x = self.patch_embed(images)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed[:, :x.shape[1]]

        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Transformer
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token 输出
        return x


# ============================================================
#  层级分类器
# ============================================================

class HierarchicalClassifier(nn.Module):
    """
    coarse → mid → fine 三级级联分类器
    
    核心思路：
    1. 先做粗粒度分类（10 个行业）
    2. 将粗粒度结果作为条件，做中粒度分类（50 个品类）
    3. 将中粒度结果作为条件，做细粒度分类（200 个子品类）
    
    条件分类通过 Embedding 注入实现
    """

    def __init__(self, config: LabelHierarchyFullConfig):
        super().__init__()
        self.config = config
        hc = config.hierarchy

        # 构建分类学树
        self.tree = TaxonomyTree(hc.num_labels_per_level)

        # 视觉编码器
        self.encoder = SimplePatchEncoder(hc)

        # 层级 Softmax
        self.hier_softmax = HierarchicalSoftmax(self.tree, hc.d_model, hc.temperature)

        # 条件嵌入：将上层预测结果编码为向量，注入下层分类
        self.level_embeddings = nn.ModuleList()
        for lv in range(hc.tree_depth):
            self.level_embeddings.append(
                nn.Embedding(hc.num_labels_per_level[lv], hc.d_model)
            )

        # 特征融合层：原始特征 + 条件嵌入
        self.fusion_layers = nn.ModuleList([
            nn.Linear(hc.d_model * 2, hc.d_model) for _ in range(hc.tree_depth - 1)
        ])

    def forward(self, images: torch.Tensor,
                target_levels: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W]
            target_levels: 每层的目标标签（局部 ID），训练时使用
        Returns:
            logits_per_level, loss
        """
        features = self.encoder(images)  # [B, D]

        # 层级 Softmax
        results = self.hier_softmax(features, target_levels)
        results['features'] = features

        return results

    def predict_with_cascade(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        级联预测：粗粒度预测结果影响细粒度
        
        与直接多级 Softmax 不同，这里利用上一层的预测结果
        通过 Embedding 注入下一层的分类
        """
        B = images.shape[0]
        features = self.encoder(images)  # [B, D]

        all_logits = []
        all_preds = []
        current_features = features

        for lv in range(self.config.hierarchy.tree_depth):
            # 当前层分类
            logits = self.hier_softmax.classifiers[lv](current_features)
            all_logits.append(logits)

            pred = logits.argmax(dim=-1)  # [B]
            all_preds.append(pred)

            # 条件注入下一层
            if lv < self.config.hierarchy.tree_depth - 1:
                cond_embed = self.level_embeddings[lv](pred)  # [B, D]
                combined = torch.cat([features, cond_embed], dim=-1)  # [B, 2D]
                current_features = self.fusion_layers[lv](combined)  # [B, D]

        return {
            'logits_per_level': all_logits,
            'predictions': all_preds,
        }


# ============================================================
#  层级约束多标签模型
# ============================================================

class ConstrainedMultiLabelModel(nn.Module):
    """
    层级约束多标签分类模型
    
    在标准多标签 BCE 基础上增加：
    1. 层级一致性约束：子节点为正 → 父节点也应为正
    2. 标签平滑：将概率质量分配给兄弟标签
    3. 分类学损失：根据树距离加权
    """

    def __init__(self, config: LabelHierarchyFullConfig):
        super().__init__()
        self.config = config
        mc = config.multi_label
        hc = config.hierarchy

        # 构建分类学树
        self.tree = TaxonomyTree(hc.num_labels_per_level)
        self.checker = LabelConsistencyChecker(self.tree)

        # 视觉编码器
        self.encoder = SimplePatchEncoder(hc)

        # 标签传播 GNN
        self.label_gnn = LabelPropagationGNN(self.tree, hc.d_model, n_layers=2)

        # 标签嵌入
        self.label_embed = nn.Embedding(self.tree.total_labels, hc.d_model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hc.d_model, hc.d_model),
            nn.ReLU(),
            nn.Dropout(hc.dropout),
            nn.Linear(hc.d_model, self.tree.total_labels),
        )

        self.penalty_weight = mc.hierarchy_penalty_weight
        self.label_smoothing = mc.label_smoothing

    def forward(self, images: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W]
            labels: [B, C] 多热标签向量
        Returns:
            predictions, loss
        """
        B = images.shape[0]
        features = self.encoder(images)  # [B, D]

        # 标签嵌入 + GNN 传播
        all_label_ids = torch.arange(self.tree.total_labels, device=images.device)
        label_feats = self.label_embed(all_label_ids)  # [N, D]
        label_feats = self.label_gnn(label_feats)  # [N, D] 传播后

        # 分类
        logits = self.classifier(features)  # [B, C]
        predictions = torch.sigmoid(logits)

        results = {
            'logits': logits,
            'predictions': predictions,
            'label_features': label_feats,
        }

        if labels is not None:
            # 标签平滑
            if self.label_smoothing > 0:
                labels = self._smooth_labels(labels)

            # BCE 损失
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

            # 层级一致性损失
            consist_loss = self._compute_consistency_loss(predictions)

            results['bce_loss'] = bce_loss
            results['consist_loss'] = consist_loss
            results['loss'] = bce_loss + self.penalty_weight * consist_loss

        return results

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """层级标签平滑：将概率质量分配给兄弟标签"""
        smoothed = labels.clone()
        eps = self.label_smoothing
        C = labels.shape[1]

        for c in range(min(C, self.tree.total_labels)):
            siblings = self.tree.get_siblings(c)
            valid_siblings = [s for s in siblings if s < C and s != c]
            if valid_siblings:
                n_siblings = len(valid_siblings)
                # 将 epsilon 的概率质量均匀分给兄弟
                smoothed[:, c] = (1 - eps) * labels[:, c] + eps / (n_siblings + 1)

        return smoothed

    def _compute_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """层级一致性损失：max(0, P(child) - P(parent))"""
        loss = torch.tensor(0.0, device=predictions.device)
        C = predictions.shape[1]
        count = 0

        for child, parent in self.tree.parent.items():
            if child < C and parent < C:
                violation = F.relu(predictions[:, child] - predictions[:, parent])
                loss = loss + violation.mean()
                count += 1

        return loss / max(count, 1)


# ============================================================
#  标签嵌入模型
# ============================================================

class LabelEmbeddingModel(nn.Module):
    """
    视觉-标签联合嵌入模型
    
    将图像和标签映射到同一嵌入空间：
    - 欧氏空间：标准余弦相似度
    - 双曲空间：Poincaré 距离（更适合层级结构）
    
    训练目标：图像嵌入靠近其正确标签，远离错误标签
    """

    def __init__(self, config: LabelHierarchyFullConfig):
        super().__init__()
        self.config = config
        ec = config.embedding
        hc = config.hierarchy

        self.use_hyperbolic = ec.use_hyperbolic
        self.margin = ec.margin
        self.curvature = ec.curvature

        # 视觉编码器
        self.encoder = SimplePatchEncoder(hc)

        # 视觉特征投影到嵌入空间
        self.visual_proj = nn.Sequential(
            nn.Linear(hc.d_model, ec.embedding_dim),
            nn.LayerNorm(ec.embedding_dim),
        )

        # 标签嵌入
        if self.use_hyperbolic:
            self.label_embed = HyperbolicEmbedding(ec)
        else:
            self.label_embed = nn.Embedding(ec.label_vocab_size, ec.embedding_dim)

    def forward(self, images: torch.Tensor,
                label_ids: torch.Tensor,
                neg_label_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W]
            label_ids: [B] 正确标签 ID
            neg_label_ids: [B, K] 负样本标签 ID（可选）
        Returns:
            visual_embed, label_embed, distance, loss
        """
        # 视觉嵌入
        features = self.encoder(images)  # [B, D]
        visual_embed = self.visual_proj(features)  # [B, E]

        # 标签嵌入
        if self.use_hyperbolic:
            label_embed = self.label_embed(label_ids)  # [B, E] 已在 Poincaré 球上
            visual_embed = exp_map_zero(visual_embed, self.curvature)
            visual_embed = project_to_ball(visual_embed, self.curvature)
            pos_dist = poincare_distance(visual_embed, label_embed)
        else:
            label_embed = self.label_embed(label_ids)
            pos_dist = 1 - F.cosine_similarity(visual_embed, label_embed)

        results = {
            'visual_embed': visual_embed,
            'label_embed': label_embed,
            'pos_distance': pos_dist,
        }

        # 对比损失
        if neg_label_ids is not None:
            B, K = neg_label_ids.shape

            if self.use_hyperbolic:
                neg_embed = self.label_embed(neg_label_ids)  # [B, K, E]
                vis_expanded = visual_embed.unsqueeze(1).expand(-1, K, -1)
                neg_dist = poincare_distance(vis_expanded, neg_embed)
            else:
                neg_embed = self.label_embed(neg_label_ids)
                vis_expanded = visual_embed.unsqueeze(1).expand(-1, K, -1)
                neg_dist = 1 - F.cosine_similarity(vis_expanded, neg_embed, dim=-1)

            # Triplet loss: max(0, d_pos - d_neg + margin)
            triplet_loss = F.relu(
                pos_dist.unsqueeze(1) - neg_dist + self.margin
            ).mean()

            results['neg_distance'] = neg_dist
            results['loss'] = triplet_loss

        return results

    def get_all_label_embeddings(self) -> torch.Tensor:
        """获取所有标签的嵌入向量"""
        all_ids = torch.arange(self.config.embedding.label_vocab_size)
        if next(self.parameters()).is_cuda:
            all_ids = all_ids.cuda()

        if self.use_hyperbolic:
            return self.label_embed(all_ids)
        else:
            return self.label_embed(all_ids)
