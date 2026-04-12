"""v04 配置"""
from dataclasses import dataclass

@dataclass
class CLIPConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    embed_dim: int = 128       # 共享 embedding 空间维度
    temperature: float = 0.07  # InfoNCE 温度系数
    vocab_size: int = 2000
    max_text_len: int = 32
    # 训练
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 30
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = CLIPConfig()
