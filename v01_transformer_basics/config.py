"""
v01 Transformer 基础 - 超参数配置

我们选择一个较小的模型配置（Mini Transformer），
确保可以在 CPU 或单 GPU 上快速训练完成，同时保留 Transformer 的所有核心组件。
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Mini Transformer 配置"""
    
    # ---- 模型架构 ----
    d_model: int = 128          # 模型隐藏维度（embedding 维度）
    n_heads: int = 4            # 多头注意力头数
    n_encoder_layers: int = 3   # 编码器层数
    n_decoder_layers: int = 3   # 解码器层数
    d_ff: int = 512             # 前馈网络中间维度（通常 4 * d_model）
    dropout: float = 0.1        # Dropout 率
    max_seq_len: int = 32       # 最大序列长度
    vocab_size: int = 32        # 词表大小（用于数字排序：0-29 + PAD + BOS/EOS）
    
    # ---- 特殊 Token ----
    pad_token_id: int = 0       # padding token
    bos_token_id: int = 30      # 序列开始 token
    eos_token_id: int = 31      # 序列结束 token
    
    # ---- 训练超参数 ----
    batch_size: int = 64        # 批大小
    learning_rate: float = 1e-3 # 学习率
    num_epochs: int = 50        # 训练轮数
    warmup_steps: int = 200     # 学习率 warmup 步数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    weight_decay: float = 0.01  # 权重衰减（L2 正则化）
    
    # ---- 数据 ----
    num_train_samples: int = 10000  # 训练样本数
    num_val_samples: int = 1000     # 验证样本数
    seq_length: int = 10            # 排序序列长度
    num_range: int = 30             # 数字范围 [0, num_range)
    
    # ---- 推理 ----
    beam_size: int = 3          # Beam Search 的 beam 大小
    
    # ---- 路径 ----
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# 默认配置实例
config = TransformerConfig()
