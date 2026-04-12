"""
V21 - 多模态数据工程配置
========================
包含数据过滤、去重、增强、课程学习、数据平衡等全流程配置。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataFilterConfig:
    """数据质量过滤配置"""
    quality_threshold: float = 0.3      # CLIP 质量评分阈值（低于此值过滤掉）
    min_resolution: int = 64            # 最小分辨率（宽和高都不低于此值）
    max_aspect_ratio: float = 3.0       # 最大宽高比（过大的图片可能是异常数据）
    min_file_size: int = 1024           # 最小文件大小（bytes），过小可能是损坏文件
    blur_threshold: float = 100.0       # 模糊度阈值（Laplacian 方差，低于此值为模糊）


@dataclass
class DedupConfig:
    """数据去重配置"""
    hash_bits: int = 128                # 哈希位数（SimHash / MinHash 签名长度）
    threshold: float = 0.8              # 相似度阈值（高于此值视为重复）
    num_perm: int = 128                 # MinHash 排列数（越大估计越准确，但更慢）
    shingle_size: int = 3               # Shingling 的 n-gram 大小
    num_bands: int = 16                 # LSH 的 band 数量（用于加速 MinHash 查询）


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    mixup_alpha: float = 0.2            # MixUp Beta 分布参数 α
    cutmix_prob: float = 0.5            # CutMix 触发概率
    cutmix_alpha: float = 1.0           # CutMix Beta 分布参数 α
    randaug_n: int = 2                  # RandAugment 每次应用的变换数
    randaug_m: int = 9                  # RandAugment 变换强度（0-30）
    color_jitter: float = 0.3           # 颜色抖动强度
    horizontal_flip_prob: float = 0.5   # 水平翻转概率


@dataclass
class CurriculumConfig:
    """课程学习配置"""
    difficulty_bins: int = 5            # 难度分级的 bin 数
    pacing_function: str = "linear"     # 节奏函数：linear / root / step
    initial_fraction: float = 0.2       # 初始阶段使用数据的比例
    warmup_epochs: int = 5              # 从简单数据过渡到全部数据的 epoch 数
    difficulty_metric: str = "loss"     # 难度衡量指标：loss / confidence / entropy


@dataclass
class FullConfig:
    """完整配置"""
    filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # 模型 & 训练
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    num_classes: int = 10

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 30
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_train_samples: int = 2000
    num_val_samples: int = 400

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
