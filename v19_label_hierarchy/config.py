"""
V19 - 层级标签理解配置
========================
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LabelHierarchyConfig:
    """层级标签分类配置"""
    tree_depth: int = 3                     # 分类树深度（粗→中→细）
    num_labels_per_level: List[int] = field(default_factory=lambda: [10, 50, 200])
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    # 视觉编码器
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # 层级 Softmax
    use_hierarchical_softmax: bool = True   # 是否使用层级 Softmax
    temperature: float = 1.0                # Softmax 温度


@dataclass
class LabelEmbeddingConfig:
    """标签嵌入配置"""
    label_vocab_size: int = 260             # 所有层级标签总数
    d_model: int = 256
    embedding_dim: int = 128                # 标签嵌入维度
    use_hyperbolic: bool = True             # 是否使用双曲空间嵌入
    curvature: float = 1.0                  # Poincaré 球曲率
    margin: float = 0.1                     # 对比学习 margin


@dataclass
class MultiLabelConfig:
    """多标签分类配置"""
    num_classes: int = 200                  # 总标签数
    threshold: float = 0.5                  # 分类阈值
    hierarchy_penalty_weight: float = 0.5   # 层级约束惩罚权重
    label_smoothing: float = 0.1            # 标签平滑系数
    max_labels_per_sample: int = 10         # 每个样本最多标签数


@dataclass
class LabelHierarchyFullConfig:
    """完整配置"""
    hierarchy: LabelHierarchyConfig = field(default_factory=LabelHierarchyConfig)
    embedding: LabelEmbeddingConfig = field(default_factory=LabelEmbeddingConfig)
    multi_label: MultiLabelConfig = field(default_factory=MultiLabelConfig)

    # 训练参数
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42

    # 数据
    num_train_samples: int = 1000
    num_val_samples: int = 200
