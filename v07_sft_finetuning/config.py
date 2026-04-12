"""v07 配置"""
from dataclasses import dataclass

@dataclass
class SFTConfig:
    # 模型 (复用 v05 架构)
    image_size: int = 64
    patch_size: int = 8
    vision_dim: int = 192
    llm_dim: int = 384
    num_queries: int = 32
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 1536
    dropout: float = 0.05
    vocab_size: int = 3000
    max_text_len: int = 64
    max_seq_len: int = 256
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"  # 注入 LoRA 的层
    # 训练
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = SFTConfig()
