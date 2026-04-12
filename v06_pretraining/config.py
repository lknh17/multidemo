"""v06 配置"""
from dataclasses import dataclass

@dataclass
class PretrainConfig:
    image_size: int = 64
    patch_size: int = 8
    vision_dim: int = 192
    llm_dim: int = 384
    num_queries: int = 32
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 1536
    dropout: float = 0.1
    vocab_size: int = 3000
    max_text_len: int = 64
    embed_dim: int = 128
    # Loss 权重
    itc_weight: float = 1.0
    itm_weight: float = 1.0
    cap_weight: float = 1.0
    temperature: float = 0.07
    # 训练
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 15
    warmup_steps: int = 100
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = PretrainConfig()
