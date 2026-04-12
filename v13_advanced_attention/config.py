"""v13 高级注意力机制 - 超参数配置"""
from dataclasses import dataclass


@dataclass
class AdvancedAttentionConfig:
    """高级注意力机制配置"""

    # ---- 模型架构 ----
    d_model: int = 256          # 模型隐藏维度
    n_heads: int = 8            # Query 头数 (MHA/GQA)
    n_kv_heads: int = 2         # KV 头数 (GQA); =n_heads 则为 MHA; =1 则为 MQA
    n_layers: int = 4           # Transformer 层数
    d_ff: int = 1024            # FFN 中间维度
    dropout: float = 0.1        # Dropout 率
    vocab_size: int = 5000      # 词表大小
    max_seq_len: int = 512      # 训练最大序列长度
    rope_theta: float = 10000.0 # RoPE 频率基数

    # ---- Sliding Window ----
    window_size: int = 128      # 滑动窗口大小
    use_sliding_window: bool = False  # 是否启用滑动窗口

    # ---- 长上下文 ----
    target_seq_len: int = 2048  # 推理目标长度 (用于 NTK 外推)
    ntk_alpha: float = 0.0      # NTK 缩放因子 (0=自动计算)

    # ---- 训练超参数 ----
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ---- 路径 ----
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


config = AdvancedAttentionConfig()
