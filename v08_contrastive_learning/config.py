"""v08 配置"""
from dataclasses import dataclass

@dataclass
class ContrastiveConfig:
    d_model: int = 256
    embed_dim: int = 128
    image_size: int = 64
    patch_size: int = 8
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    vocab_size: int = 3000
    max_text_len: int = 32
    temperature: float = 0.07
    margin: float = 0.2
    # 训练
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = ContrastiveConfig()
