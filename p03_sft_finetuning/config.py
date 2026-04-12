"""
p03 SFT 指令微调 - 配置文件

管理 SFT 指令微调的所有超参数，支持 4 种微调方法：
1. full_finetune — 全参微调
2. lora — LoRA 微调
3. qlora — QLoRA 微调（4-bit 量化 + LoRA）
4. dora — DoRA 微调（权重分解 + LoRA）

包含 24G/48G 两套 GPU preset，以及 LoRA rank/alpha 消融实验配置。

使用方式:
    from config import config, config_48g, lora_config, ablation_configs
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. SFT 训练配置
# ============================================================
@dataclass
class SFTConfig:
    """SFT 指令微调配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B"      # 基座模型
    method: str = "lora"                          # 微调方法: full_finetune / lora / qlora / dora
    
    # ---- 数据 ----
    data_dir: str = "data"                        # 数据目录
    dataset_format: str = "alpaca"                # 数据格式: alpaca / sharegpt
    max_seq_length: int = 512                     # 最大序列长度
    max_samples: int = 50000                      # 最大训练样本数
    val_ratio: float = 0.05                       # 验证集比例
    
    # ---- 训练超参数 ----
    num_train_epochs: int = 3                     # 训练轮数（SFT 通常 1-5 轮）
    per_device_train_batch_size: int = 4          # 每设备 batch size
    per_device_eval_batch_size: int = 4           # 每设备评估 batch size
    gradient_accumulation_steps: int = 4          # 梯度累积步数（等效 batch = 4×4 = 16）
    learning_rate: float = 2e-4                   # 学习率（LoRA 用较大的 lr）
    lr_scheduler_type: str = "cosine"             # 学习率策略
    warmup_ratio: float = 0.05                    # Warmup 比例
    weight_decay: float = 0.01                    # 权重衰减
    max_grad_norm: float = 1.0                    # 梯度裁剪
    
    # ---- 精度 & 优化 ----
    bf16: bool = True                             # 使用 bf16 混合精度
    gradient_checkpointing: bool = True           # 梯度检查点
    
    # ---- DeepSpeed ----
    deepspeed_config: Optional[str] = None        # DeepSpeed 配置文件路径
    
    # ---- 保存 & 日志 ----
    output_dir: str = "outputs/sft"
    logging_steps: int = 10                       # 每 N 步打印一次 loss
    save_steps: int = 200                         # 每 N 步保存 checkpoint
    save_total_limit: int = 3                     # 最多保留 N 个 checkpoint
    eval_steps: int = 100                         # 每 N 步评估一次
    eval_strategy: str = "steps"                  # 评估策略: steps / epoch
    
    # ---- 路径 ----
    seed: int = 42
    log_dir: str = "logs/sft"


# ============================================================
# 2. LoRA 配置
# ============================================================
@dataclass
class LoRAConfig:
    """LoRA / QLoRA / DoRA 的共用超参数"""
    
    # ---- LoRA 基本参数 ----
    lora_r: int = 16                              # LoRA 秩（SFT 通常 8-64）
    lora_alpha: int = 32                          # LoRA alpha（通常 = 2 × r）
    lora_dropout: float = 0.05                    # Dropout 防过拟合
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",      # FFN 层
    ])
    
    # ---- QLoRA 专用 ----
    load_in_4bit: bool = False                    # 是否 4-bit 量化加载
    bnb_4bit_compute_dtype: str = "bfloat16"      # 量化计算精度
    bnb_4bit_quant_type: str = "nf4"              # 量化类型: nf4 / fp4
    bnb_4bit_use_double_quant: bool = True        # 双重量化（进一步压缩）
    
    # ---- DoRA 专用 ----
    use_dora: bool = False                        # 是否启用 DoRA（权重分解）
    
    # ---- 学习率（LoRA 系列通常用较大的 lr）----
    learning_rate: float = 2e-4                   # LoRA/QLoRA 学习率
    
    # ---- 全参微调的学习率（更小）----
    full_finetune_lr: float = 2e-5                # 全参微调学习率


# ============================================================
# 3. 消融实验配置
# ============================================================
@dataclass
class AblationConfig:
    """消融实验配置：用于 ablation_runner.py"""
    
    # LoRA rank 消融
    lora_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    
    # LoRA alpha 消融（alpha / rank 的比值）
    alpha_ratios: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    
    # target_modules 消融
    target_module_groups: dict = field(default_factory=lambda: {
        "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "qv_only": ["q_proj", "v_proj"],
        "attn_ffn": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    })
    
    # 每组实验的训练步数（消融不需要完整训练）
    max_steps: int = 500
    eval_steps: int = 50
    save_results_path: str = "outputs/ablation_results.json"


# ============================================================
# 4. GPU Preset
# ============================================================
def create_config_24g() -> SFTConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = SFTConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4
    cfg.max_seq_length = 512
    cfg.gradient_checkpointing = True
    return cfg


def create_config_48g() -> SFTConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = SFTConfig()
    cfg.per_device_train_batch_size = 8
    cfg.gradient_accumulation_steps = 2
    cfg.max_seq_length = 1024
    cfg.gradient_checkpointing = False  # 48G 显存充足，不需要 GC
    return cfg


# ============================================================
# 5. 快捷方法配置创建
# ============================================================
def create_method_config(method: str) -> tuple:
    """
    根据方法名创建对应配置。
    
    Args:
        method: full_finetune / lora / qlora / dora
    
    Returns:
        (sft_config, lora_config)
    """
    cfg = SFTConfig()
    lcfg = LoRAConfig()
    
    cfg.method = method
    
    if method == "full_finetune":
        # 全参微调：不用 LoRA，更小学习率
        cfg.learning_rate = lcfg.full_finetune_lr
        cfg.output_dir = "outputs/sft_full"
    
    elif method == "lora":
        # 标准 LoRA
        cfg.learning_rate = lcfg.learning_rate
        cfg.output_dir = "outputs/sft_lora"
    
    elif method == "qlora":
        # QLoRA：4-bit 量化 + LoRA
        lcfg.load_in_4bit = True
        cfg.learning_rate = lcfg.learning_rate
        cfg.output_dir = "outputs/sft_qlora"
    
    elif method == "dora":
        # DoRA：权重分解 + LoRA
        lcfg.use_dora = True
        cfg.learning_rate = lcfg.learning_rate
        cfg.output_dir = "outputs/sft_dora"
    
    return cfg, lcfg


# 默认配置
config = SFTConfig()
config_48g = create_config_48g()
lora_config = LoRAConfig()
ablation_config = AblationConfig()
