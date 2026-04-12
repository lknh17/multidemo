"""
p05 强化学习 GRPO - 配置文件

管理 GRPO 训练的所有超参数，包含奖励函数权重、KL 约束、
Group 采样参数等。支持多组消融实验配置。

使用方式:
    from config import config, ablation_configs
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. GRPO 训练配置
# ============================================================
@dataclass
class GRPOConfig:
    """GRPO 强化学习训练配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"   # SFT 后的模型（或用 DPO 后的）
    ref_model_name: Optional[str] = None               # 参考模型（默认同 model_name）
    sft_model_path: Optional[str] = None               # SFT 模型路径（用于对比）
    dpo_model_path: Optional[str] = None               # DPO 模型路径（用于对比）
    
    # ---- 数据 ----
    dataset_name: str = "openai/gsm8k"                 # GSM8K 数学推理数据集
    dataset_split: str = "train"
    max_samples: int = 5000                             # 训练样本数
    eval_samples: int = 500                             # 评估样本数
    
    # ---- GRPO 核心参数 ----
    group_size: int = 8                                 # 每个 prompt 采样的响应数量 G
    temperature: float = 0.7                            # 采样温度
    top_p: float = 0.95                                 # Top-p 采样
    max_new_tokens: int = 512                           # 最大生成长度
    
    # ---- 策略优化 ----
    clip_ratio: float = 0.2                             # PPO clip 范围 ε
    kl_coef: float = 0.05                               # KL 散度惩罚系数 β
    kl_target: Optional[float] = None                   # KL 目标值（自适应 KL）
    entropy_coef: float = 0.01                          # 熵正则化系数
    
    # ---- 奖励函数权重 ----
    reward_correctness_weight: float = 0.6              # 正确性奖励权重
    reward_format_weight: float = 0.3                   # 格式奖励权重
    reward_length_weight: float = 0.1                   # 长度惩罚权重
    
    # ---- 训练超参数 ----
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2                # GRPO 显存消耗大，batch 较小
    gradient_accumulation_steps: int = 8                # 等效 batch = 2×8 = 16
    learning_rate: float = 5e-7                         # RL 训练用非常小的学习率
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ---- 精度 & 优化 ----
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # ---- 保存 & 日志 ----
    output_dir: str = "outputs/grpo"
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
    
    # ---- 路径 ----
    seed: int = 42
    log_dir: str = "logs/grpo"
    
    # ---- 奖励函数选择 ----
    reward_type: str = "composite"                      # correctness / format / composite


# ============================================================
# 2. 奖励函数配置
# ============================================================
@dataclass
class RewardConfig:
    """奖励函数详细配置"""
    
    # ---- 正确性奖励 ----
    correct_reward: float = 1.0                         # 正确答案奖励
    incorrect_reward: float = -0.5                      # 错误答案惩罚
    partial_reward: float = 0.3                         # 部分正确奖励（格式对但数值错）
    no_answer_reward: float = -1.0                      # 无法提取答案惩罚
    
    # ---- 格式奖励 ----
    has_steps_reward: float = 0.3                       # 包含推理步骤
    has_final_answer_reward: float = 0.2                # 包含最终答案标记
    step_count_min: int = 2                             # 最少推理步骤数
    step_count_max: int = 10                            # 最多推理步骤数
    
    # ---- 长度惩罚 ----
    max_response_length: int = 500                      # 超过此长度开始惩罚
    length_penalty_factor: float = 0.001                # 每多一个 token 的惩罚


# ============================================================
# 3. 消融实验配置
# ============================================================
def create_ablation_group_size(g: int) -> GRPOConfig:
    """消融实验：不同 group_size"""
    cfg = GRPOConfig()
    cfg.group_size = g
    cfg.output_dir = f"outputs/grpo_g{g}"
    cfg.log_dir = f"logs/grpo_g{g}"
    return cfg


def create_ablation_kl_coef(beta: float) -> GRPOConfig:
    """消融实验：不同 KL 系数"""
    cfg = GRPOConfig()
    cfg.kl_coef = beta
    cfg.output_dir = f"outputs/grpo_kl{beta}"
    cfg.log_dir = f"logs/grpo_kl{beta}"
    return cfg


def create_ablation_reward_type(rtype: str) -> GRPOConfig:
    """消融实验：不同奖励函数"""
    cfg = GRPOConfig()
    cfg.reward_type = rtype
    if rtype == "correctness":
        cfg.reward_correctness_weight = 1.0
        cfg.reward_format_weight = 0.0
        cfg.reward_length_weight = 0.0
    elif rtype == "format":
        cfg.reward_correctness_weight = 0.0
        cfg.reward_format_weight = 1.0
        cfg.reward_length_weight = 0.0
    cfg.output_dir = f"outputs/grpo_{rtype}"
    cfg.log_dir = f"logs/grpo_{rtype}"
    return cfg


def create_config_24g() -> GRPOConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = GRPOConfig()
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.group_size = 4                                  # 显存不够用更小的 group
    cfg.max_new_tokens = 384
    cfg.gradient_checkpointing = True
    return cfg


def create_config_48g() -> GRPOConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = GRPOConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4
    cfg.group_size = 16                                 # 更大的 group 更稳定
    cfg.max_new_tokens = 512
    cfg.gradient_checkpointing = False
    return cfg


# ============================================================
# 4. 默认实例
# ============================================================
config = GRPOConfig()
config_24g = create_config_24g()
config_48g = create_config_48g()
reward_config = RewardConfig()

# 消融实验预设
ablation_configs = {
    "group_4": create_ablation_group_size(4),
    "group_8": create_ablation_group_size(8),
    "group_16": create_ablation_group_size(16),
    "kl_0.01": create_ablation_kl_coef(0.01),
    "kl_0.05": create_ablation_kl_coef(0.05),
    "kl_0.1": create_ablation_kl_coef(0.1),
    "reward_correctness": create_ablation_reward_type("correctness"),
    "reward_format": create_ablation_reward_type("format"),
    "reward_composite": create_ablation_reward_type("composite"),
}
