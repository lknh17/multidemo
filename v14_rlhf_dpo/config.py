"""v14 RLHF / DPO 偏好对齐 - 超参数配置"""
from dataclasses import dataclass


@dataclass
class RLHFConfig:
    """RLHF / DPO 偏好对齐配置"""

    # ---- 策略模型 (Policy) ----
    d_model: int = 256          # 模型隐藏维度
    n_heads: int = 8            # 注意力头数
    n_layers: int = 4           # Transformer 层数
    d_ff: int = 1024            # FFN 中间维度
    dropout: float = 0.1
    vocab_size: int = 5000      # 词表大小
    max_seq_len: int = 256      # 最大序列长度

    # ---- Reward Model ----
    reward_n_layers: int = 4    # Reward Model 层数
    reward_head_dim: int = 128  # Reward head 隐藏维度

    # ---- DPO 超参数 ----
    dpo_beta: float = 0.1       # DPO 温度参数 (β)
                                 # β 越大 → 越保守，偏离 ref 模型越少
                                 # β 越小 → 越激进，更大幅度偏好优化
    label_smoothing: float = 0.0 # DPO label smoothing

    # ---- SimPO 超参数 ----
    simpo_gamma: float = 0.5    # SimPO 奖励边际
    simpo_beta: float = 2.0     # SimPO 温度

    # ---- KTO 超参数 ----
    kto_beta: float = 0.1       # KTO 温度
    kto_desirable_weight: float = 1.0    # 正样本权重
    kto_undesirable_weight: float = 1.0  # 负样本权重

    # ---- PPO 超参数 ----
    ppo_clip_ratio: float = 0.2       # PPO 裁剪范围
    ppo_value_clip: float = 0.2       # Value function 裁剪
    ppo_kl_coef: float = 0.02         # KL 惩罚系数
    ppo_epochs: int = 4               # 每批数据的 PPO 迭代次数
    gae_lambda: float = 0.95          # GAE 参数

    # ---- 训练超参数 ----
    batch_size: int = 16
    learning_rate: float = 1e-5       # 对齐阶段 LR 远低于预训练
    num_epochs: int = 3               # 对齐通常只需 1-3 epoch
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # ---- 路径 ----
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


config = RLHFConfig()
