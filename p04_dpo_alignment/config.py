"""
p04 DPO 对齐训练 - 配置文件

管理 DPO/SimPO/ORPO/KTO 四种对齐算法的超参数，
包含 beta 消融实验设置、24G/48G 两套 GPU preset。

使用方式:
    from config import config, config_48g
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. DPO 对齐训练配置
# ============================================================
@dataclass
class DPOConfig:
    """DPO / SimPO / ORPO / KTO 对齐训练配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # SFT 后的模型（或 Instruct 版）
    ref_model_name: Optional[str] = None               # 参考模型（None=复制 policy）
    
    # ---- 算法选择 ----
    algorithm: str = "dpo"                             # dpo / simpo / orpo / kto
    
    # ---- DPO 超参数 ----
    beta: float = 0.1                                  # KL 散度惩罚系数
    label_smoothing: float = 0.0                       # 标签平滑（防止过拟合）
    loss_type: str = "sigmoid"                         # DPO loss 类型: sigmoid / hinge / ipo
    
    # ---- SimPO 超参数 ----
    simpo_gamma: float = 0.5                           # SimPO 的 margin 参数 γ
    cpo_alpha: float = 1.0                             # CPO/SimPO 中 NLL loss 的权重
    
    # ---- ORPO 超参数 ----
    orpo_alpha: float = 1.0                            # ORPO odds ratio loss 权重
    
    # ---- KTO 超参数 ----
    kto_desirable_weight: float = 1.0                  # KTO 对 chosen 的权重
    kto_undesirable_weight: float = 1.0                # KTO 对 rejected 的权重
    
    # ---- Beta 消融实验 ----
    beta_ablation_values: List[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.1, 0.2, 0.5, 1.0
    ])
    
    # ---- 数据 ----
    dataset_name: str = "ultrafeedback"                # 数据集: ultrafeedback / hh_rlhf / custom
    data_path: str = "data/preference.jsonl"           # 本地数据路径
    max_samples: int = 10000                           # 最大训练样本数
    max_prompt_length: int = 512                       # prompt 最大 token 数
    max_length: int = 1024                             # prompt + response 最大 token 数
    
    # ---- 训练超参数 ----
    num_train_epochs: int = 1                          # 训练轮数（DPO 通常 1-3 轮）
    per_device_train_batch_size: int = 2               # 每设备 batch size（偏好对占显存大）
    gradient_accumulation_steps: int = 8               # 梯度累积（等效 batch = 2×8 = 16）
    learning_rate: float = 5e-7                        # 学习率（DPO 用非常小的 lr）
    lr_scheduler_type: str = "cosine"                  # 学习率策略
    warmup_ratio: float = 0.1                          # Warmup 比例
    weight_decay: float = 0.01                         # 权重衰减
    max_grad_norm: float = 1.0                         # 梯度裁剪
    
    # ---- LoRA ----
    use_lora: bool = True                              # 是否使用 LoRA（推荐）
    lora_r: int = 16                                   # LoRA 秩
    lora_alpha: int = 32                               # LoRA alpha
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # ---- 精度 & 优化 ----
    bf16: bool = True                                  # bf16 混合精度
    gradient_checkpointing: bool = True                # 梯度检查点
    
    # ---- 保存 & 日志 ----
    output_dir: str = "outputs/dpo"
    logging_steps: int = 5                             # 每 N 步打印 loss
    save_steps: int = 200                              # 每 N 步保存 checkpoint
    save_total_limit: int = 3                          # 最多保留 N 个 checkpoint
    eval_steps: int = 100                              # 每 N 步评估
    eval_ratio: float = 0.05                           # 验证集比例
    
    # ---- 路径 ----
    seed: int = 42
    log_dir: str = "logs/dpo"


# ============================================================
# 2. 算法特定默认值
# ============================================================
ALGORITHM_DEFAULTS = {
    "dpo": {
        "beta": 0.1,
        "learning_rate": 5e-7,
        "loss_type": "sigmoid",
    },
    "simpo": {
        "beta": 2.0,                # SimPO 论文推荐 beta=2.0
        "learning_rate": 1e-6,
        "simpo_gamma": 0.5,
        "loss_type": "sigmoid",
    },
    "orpo": {
        "beta": 0.1,                # ORPO 不依赖 ref model，beta 主要用于 log ratio
        "learning_rate": 5e-6,       # ORPO 可以用更大的 lr
        "orpo_alpha": 1.0,
    },
    "kto": {
        "beta": 0.1,
        "learning_rate": 5e-7,
        "kto_desirable_weight": 1.0,
        "kto_undesirable_weight": 1.0,
    },
}


def apply_algorithm_defaults(cfg: DPOConfig) -> DPOConfig:
    """根据算法类型应用默认超参数"""
    defaults = ALGORITHM_DEFAULTS.get(cfg.algorithm, {})
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


# ============================================================
# 3. GPU Preset
# ============================================================
def create_config_24g() -> DPOConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = DPOConfig()
    cfg.per_device_train_batch_size = 2
    cfg.gradient_accumulation_steps = 8
    cfg.max_length = 1024
    cfg.max_prompt_length = 512
    cfg.gradient_checkpointing = True
    cfg.use_lora = True                                # 24G 必须用 LoRA
    return cfg


def create_config_48g() -> DPOConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = DPOConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4
    cfg.max_length = 2048
    cfg.max_prompt_length = 1024
    cfg.gradient_checkpointing = False
    cfg.use_lora = True                                # LoRA 仍推荐（加速训练）
    return cfg


# 默认配置
config = DPOConfig()
config_48g = create_config_48g()
