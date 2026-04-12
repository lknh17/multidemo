"""v05 配置"""
from dataclasses import dataclass

@dataclass
class QwenVLConfig:
    # Visual Resampler
    vision_dim: int = 256      # 视觉编码器输出维度（demo 缩小版）
    llm_dim: int = 512         # LLM 隐藏维度
    num_queries: int = 64      # Resampler 查询数量（压缩后的 token 数）
    num_vision_tokens: int = 196  # ViT 输出的 token 数
    resampler_heads: int = 8
    resampler_layers: int = 3
    resampler_ff_dim: int = 1024
    dropout: float = 0.1
    # 模型
    vocab_size: int = 5000
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    max_seq_len: int = 512
    image_size: int = 224
    patch_size: int = 16
    # 训练
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = QwenVLConfig()
