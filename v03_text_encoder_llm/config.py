"""v03 配置"""
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 5000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    # 训练
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 20
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    # 推理
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_gen_len: int = 100
    # 路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = GPTConfig()
