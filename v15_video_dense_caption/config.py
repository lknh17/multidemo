"""
V15 - 视频理解与 Dense Captioning 配置
======================================
涵盖视频编码器、Dense Video Captioning、时序 Grounding 三大模块配置
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VideoEncoderConfig:
    """视频编码器配置"""
    # 输入参数
    num_frames: int = 16                # 采样帧数
    frame_size: int = 224               # 帧分辨率
    patch_size: int = 16                # ViT patch 大小
    in_channels: int = 3                # 输入通道数

    # 模型参数
    d_model: int = 256                  # 隐藏维度
    n_heads: int = 8                    # 注意力头数
    n_layers: int = 4                   # Transformer 层数
    d_ff: int = 1024                    # FFN 中间维度
    dropout: float = 0.1

    # 时序建模
    temporal_model: str = "timesformer"  # timesformer / video_swin / conv3d
    temporal_kernel_size: int = 3        # 3D Conv 时序核大小
    temporal_stride: int = 1


@dataclass
class DenseCaptionConfig:
    """Dense Video Captioning 配置"""
    # 时序提议网络
    num_proposals: int = 100            # 候选提议数量
    proposal_hidden_dim: int = 256      # 提议网络隐藏维度
    nms_threshold: float = 0.5          # 时序 NMS 阈值
    min_proposal_length: float = 0.1    # 最小提议长度（占视频比例）
    max_proposal_length: float = 0.8    # 最大提议长度

    # 描述生成器
    caption_d_model: int = 256          # Caption 解码器维度
    caption_n_heads: int = 8
    caption_n_layers: int = 3
    caption_vocab_size: int = 5000      # 词表大小
    max_caption_len: int = 30           # 最大描述长度
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Deformable 时序注意力
    n_deform_points: int = 4            # 可变形采样点数
    n_deform_levels: int = 3            # 多尺度层级数

    # 损失权重
    proposal_cls_weight: float = 1.0
    proposal_reg_weight: float = 5.0
    caption_weight: float = 1.0


@dataclass
class TemporalGroundingConfig:
    """时序 Grounding 配置"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    text_d_model: int = 256             # 文本编码器维度
    max_text_len: int = 50
    vocab_size: int = 5000

    # Grounding 输出
    num_moment_queries: int = 10        # Moment Query 数量
    span_loss_weight: float = 5.0
    cls_loss_weight: float = 1.0


@dataclass
class VideoDenseCaptionFullConfig:
    """完整配置"""
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    dense_caption: DenseCaptionConfig = field(default_factory=DenseCaptionConfig)
    temporal_grounding: TemporalGroundingConfig = field(default_factory=TemporalGroundingConfig)

    # 训练参数
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 30
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42

    # 数据参数
    num_train_videos: int = 500
    num_val_videos: int = 100
    video_max_seconds: float = 30.0     # 视频最长秒数
    fps: float = 2.0                    # 采样 FPS
