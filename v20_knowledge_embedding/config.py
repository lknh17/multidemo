"""
V20 - 知识增强多模态嵌入配置
============================
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KnowledgeGraphConfig:
    """知识图谱嵌入配置（TransE / TransR）"""
    num_entities: int = 5000            # 实体数量
    num_relations: int = 200            # 关系数量
    d_model: int = 256                  # 嵌入维度
    n_heads: int = 8                    # GNN 注意力头数
    n_gnn_layers: int = 2              # GNN 层数
    margin: float = 1.0                 # TransE margin
    norm_p: int = 2                     # L_p 距离范数（1 或 2）
    relation_space_dim: int = 128       # TransR 关系空间维度
    negative_samples: int = 10          # 负采样数量


@dataclass
class KGEnhancedEmbeddingConfig:
    """KG 增强嵌入配置"""
    visual_dim: int = 256               # 视觉特征维度
    text_dim: int = 256                 # 文本特征维度
    kg_dim: int = 256                   # KG 嵌入维度
    fusion_dim: int = 256               # 融合后维度
    n_heads: int = 8                    # 注意力头数
    n_layers: int = 4                   # Transformer 层数
    d_ff: int = 1024                    # FFN 维度
    dropout: float = 0.1
    fusion_strategy: str = "attention"  # 融合策略：attention / concat / gate
    image_size: int = 224               # 输入图像尺寸
    patch_size: int = 16                # ViT patch 大小
    in_channels: int = 3
    max_entities_per_image: int = 10    # 每张图最多关联实体数


@dataclass
class EntityLinkConfig:
    """实体链接配置"""
    d_model: int = 256
    mention_hidden_dim: int = 128       # mention 检测隐藏维度
    max_mention_len: int = 16           # 最大 mention 长度
    max_candidates: int = 20            # 每个 mention 最多候选实体数
    vocab_size: int = 5000              # 文本词表大小
    max_seq_len: int = 128              # 最大序列长度
    link_threshold: float = 0.5         # 链接置信度阈值


@dataclass
class KnowledgeEmbeddingFullConfig:
    """完整配置"""
    kg: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    kg_embed: KGEnhancedEmbeddingConfig = field(default_factory=KGEnhancedEmbeddingConfig)
    entity_link: EntityLinkConfig = field(default_factory=EntityLinkConfig)

    # 训练参数
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42

    # 知识蒸馏
    distill_temperature: float = 4.0    # 蒸馏温度
    distill_alpha: float = 0.5          # 蒸馏损失权重

    # 数据
    num_train_samples: int = 1000
    num_val_samples: int = 200
    num_kg_triplets: int = 10000        # KG 三元组数量
