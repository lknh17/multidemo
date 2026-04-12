"""
V25 - 端到端广告多模态管线配置（Capstone）
============================================
整合 V01-V24 所有模块，构建完整广告投放管线。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class AdCreativeConfig:
    """广告创意编码配置"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    image_size: int = 224          # 视觉编码输入尺寸
    patch_size: int = 16
    in_channels: int = 3
    text_max_len: int = 64         # 文本最大 token 数
    vocab_size: int = 5000         # 文本词汇表大小
    has_audio: bool = True         # 是否包含音频模态
    has_video: bool = False        # 是否包含视频模态
    audio_dim: int = 128           # 音频特征维度
    fusion_type: str = "attention" # 融合方式: attention / concat / gated


@dataclass
class PipelineConfig:
    """管线阶段配置"""
    stages: List[str] = field(default_factory=lambda: [
        "encode", "retrieve", "rerank", "safety_filter", "quality_gate", "serve"
    ])
    enable_safety: bool = True     # 启用安全过滤
    enable_quality: bool = True    # 启用质量门控
    cache_embeddings: bool = True  # 缓存已计算的 embedding
    embedding_dim: int = 256       # 统一 embedding 维度
    index_type: str = "flat"       # ANN 索引类型: flat / ivf / hnsw


@dataclass
class MatchingConfig:
    """检索匹配配置"""
    retrieval_top_k: int = 100     # 召回阶段取 top-K
    rerank_top_k: int = 20         # 精排阶段取 top-K
    reranker_type: str = "cross"   # 重排器类型: cross / poly / colbert
    similarity_metric: str = "cosine"  # 相似度度量
    ann_nprobe: int = 8            # IVF 索引探测数


@dataclass
class MonitorConfig:
    """监控告警配置"""
    metrics_window: int = 100      # 滑动窗口大小
    alert_threshold: float = 0.05  # 告警阈值（如 CTR 下降 5%）
    log_interval: int = 10         # 日志间隔
    enable_ab_test: bool = True    # 启用 A/B 实验


@dataclass
class FullConfig:
    """完整配置 — 整合全部子模块"""
    creative: AdCreativeConfig = field(default_factory=AdCreativeConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    # 训练超参
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_train_samples: int = 2000
    num_val_samples: int = 400

    # 多目标权重
    ctr_weight: float = 0.4
    relevance_weight: float = 0.3
    diversity_weight: float = 0.2
    freshness_weight: float = 0.1
