"""
V17 - 音频理解与全模态模型配置
==============================
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MelSpectrogramConfig:
    """Mel 频谱配置"""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 160           # 10ms
    n_mels: int = 128               # Mel 滤波器数量
    max_duration: float = 10.0       # 最大音频时长（秒）
    f_min: float = 0.0
    f_max: float = 8000.0


@dataclass
class AudioEncoderConfig:
    """音频编码器配置（AST 风格）"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    n_mels: int = 128               # 频率维度
    time_patch_size: int = 16       # 时间维度 patch 大小
    freq_patch_size: int = 16       # 频率维度 patch 大小
    max_time_patches: int = 64      # 最大时间 patch 数


@dataclass
class CLAPConfig:
    """CLAP（音频-文本对齐）配置"""
    d_model: int = 256
    audio_d_model: int = 256
    text_d_model: int = 256
    projection_dim: int = 128
    temperature: float = 0.07
    vocab_size: int = 5000
    max_text_len: int = 77


@dataclass
class OmniModalConfig:
    """全模态融合配置"""
    d_model: int = 256
    n_heads: int = 8
    n_fusion_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    # 各模态编码器输出维度
    image_dim: int = 256
    text_dim: int = 256
    audio_dim: int = 256

    # 图像
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # 文本
    vocab_size: int = 5000
    max_text_len: int = 128

    # 音频
    n_mels: int = 128
    max_time_patches: int = 64

    # 融合
    num_query_tokens: int = 32       # Q-Former 查询 token 数
    num_classes: int = 20            # 分类任务类别数


@dataclass
class AudioMultimodalFullConfig:
    """完整配置"""
    mel: MelSpectrogramConfig = field(default_factory=MelSpectrogramConfig)
    audio_enc: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    clap: CLAPConfig = field(default_factory=CLAPConfig)
    omni: OmniModalConfig = field(default_factory=OmniModalConfig)

    # 训练参数
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42

    # 数据
    num_train_samples: int = 1000
    num_val_samples: int = 200
