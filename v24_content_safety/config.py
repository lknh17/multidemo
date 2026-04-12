"""
V24 - 内容安全与合规配置
========================
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SafetyClassifierConfig:
    """内容安全分类器配置"""
    num_categories: int = 8             # NSFW/暴力/垃圾/仇恨/自残/恐怖/违禁/正常
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    threshold: float = 0.5             # 安全阈值
    label_names: List[str] = None

    def __post_init__(self):
        if self.label_names is None:
            self.label_names = [
                'nsfw', 'violence', 'spam', 'hate',
                'self_harm', 'terrorism', 'contraband', 'normal'
            ]


@dataclass
class ToxicityConfig:
    """毒性检测配置"""
    text_model_dim: int = 256
    image_model_dim: int = 256
    fused_dim: int = 256
    n_heads: int = 8
    vocab_size: int = 5000
    max_seq_len: int = 128
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_toxicity_dims: int = 6          # 毒性维度：侮辱/威胁/淫秽/歧视/仇恨/正常


@dataclass
class WatermarkConfig:
    """水印嵌入配置"""
    watermark_bits: int = 32            # 水印比特数
    d_model: int = 256
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    robustness_level: float = 0.5       # 鲁棒性强度 [0, 1]
    alpha: float = 0.1                  # 水印嵌入强度


@dataclass
class AdversarialConfig:
    """对抗鲁棒性配置"""
    d_model: int = 256
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 8
    epsilon: float = 8.0 / 255.0       # 扰动上界 (L∞)
    attack_steps: int = 10              # PGD 迭代步数
    step_size: float = 2.0 / 255.0     # PGD 步长
    adversarial_ratio: float = 0.5     # 对抗样本比例


@dataclass
class SafetyFullConfig:
    """完整配置"""
    safety_cls: SafetyClassifierConfig = field(default_factory=SafetyClassifierConfig)
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)

    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_train_samples: int = 1000
    num_val_samples: int = 200
