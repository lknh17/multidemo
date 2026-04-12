"""
V23 - 在线推理服务系统配置
==========================
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ServingConfig:
    """服务部署配置"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 100
    max_batch_size: int = 32           # 动态批处理最大批大小
    max_latency_ms: float = 50.0       # 最大容忍延迟 (ms)
    num_workers: int = 4               # 推理工作线程数


@dataclass
class ExportConfig:
    """模型导出配置"""
    export_format: str = "onnx"        # onnx / torchscript
    quantize: bool = True              # 是否进行 INT8 量化
    opset_version: int = 17            # ONNX opset 版本
    dynamic_axes: bool = True          # 是否支持动态 batch 维度
    optimize_level: int = 2            # 优化等级 (0-3)


@dataclass
class CacheConfig:
    """模型缓存配置"""
    cache_size: int = 1024             # 缓存条目数
    ttl_seconds: float = 300.0         # 缓存过期时间 (秒)
    eviction_policy: str = "lru"       # 缓存淘汰策略: lru / lfu / fifo


@dataclass
class IndexConfig:
    """向量索引配置"""
    embedding_dim: int = 256           # 向量维度
    index_type: str = "ivf"            # 索引类型: flat / ivf / hnsw
    nprobe: int = 16                   # IVF 搜索时探测的簇数
    nlist: int = 256                   # IVF 聚类中心数
    top_k: int = 10                    # 检索返回 Top-K


@dataclass
class FullConfig:
    """完整配置"""
    serving: ServingConfig = field(default_factory=ServingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_train_samples: int = 500
    num_val_samples: int = 100
