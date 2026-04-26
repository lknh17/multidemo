"""
p02 继续预训练 - 配置文件

管理继续预训练的所有超参数，包含 24G/48G 两套 GPU preset，
支持多种训练策略（全参/LoRA）和学习率策略（cosine/linear）的消融实验。

使用方式:
    from config import config, config_48g
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 预训练配置
# ============================================================
@dataclass
class PretrainConfig:
    """继续预训练配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B"      # 基座模型
    
    # ---- 数据 ----
    dataset_name: str = "wikimedia/wikipedia"     # 数据集名称（新版）
    dataset_config: str = "20231101.zh"           # Wikipedia 中文子集
    dataset_split: str = "train"
    max_samples: int = 10000                      # 长文本 10000 条已足够（优于 50000 条短句）
    streaming: bool = True                        # 流式加载（节省内存）
    
    # ---- Tokenize & Packing ----
    max_seq_length: int = 512                     # 最大序列长度
    packing: bool = True                          # 是否启用 packing（将多条短文本拼接为一条长序列）
    
    # ---- 训练超参数 ----
    num_train_epochs: int = 1                     # 训练轮数（继续预训练通常 1-3 轮）
    per_device_train_batch_size: int = 4          # 每设备 batch size
    gradient_accumulation_steps: int = 4          # 梯度累积步数（等效 batch = 4×4 = 16）
    learning_rate: float = 5e-6                   # 学习率（降低到 5e-6，防止小模型灾难性遗忘）
    lr_scheduler_type: str = "cosine"             # 学习率策略: cosine / linear / constant
    warmup_ratio: float = 0.05                    # Warmup 比例（总步数的 5%）
    weight_decay: float = 0.01                    # 权重衰减
    max_grad_norm: float = 1.0                    # 梯度裁剪
    
    # ---- 精度 & 优化 ----
    bf16: bool = True                             # 使用 bf16 混合精度
    gradient_checkpointing: bool = True           # 梯度检查点（用时间换空间）
    
    # ---- DeepSpeed ----
    deepspeed_config: Optional[str] = None        # DeepSpeed 配置文件路径
    
    # ---- 保存 & 日志 ----
    output_dir: str = "outputs/pretrain"
    logging_steps: int = 10                       # 每 N 步打印一次 loss
    save_steps: int = 500                         # 每 N 步保存 checkpoint
    save_total_limit: int = 3                     # 最多保留 N 个 checkpoint
    eval_steps: int = 200                         # 每 N 步评估一次
    
    # ---- 路径 ----
    seed: int = 42
    log_dir: str = "logs/pretrain"


# ============================================================
# 2. LoRA 预训练配置
# ============================================================
@dataclass
class LoRAPretrainConfig:
    """LoRA 预训练的额外配置"""
    
    lora_r: int = 64                              # LoRA 秩（预训练用较大的 rank）
    lora_alpha: int = 128                         # LoRA alpha（通常 = 2 × r）
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",      # FFN 层
    ])
    # LoRA 预训练使用更大的学习率
    learning_rate: float = 1e-4


# ============================================================
# 3. GPU Preset
# ============================================================
def create_config_24g() -> PretrainConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = PretrainConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4
    cfg.max_seq_length = 512
    cfg.gradient_checkpointing = True
    return cfg


def create_config_48g() -> PretrainConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = PretrainConfig()
    cfg.per_device_train_batch_size = 8
    cfg.gradient_accumulation_steps = 2
    cfg.max_seq_length = 1024
    cfg.gradient_checkpointing = False  # 48G 显存充足，不需要 GC
    return cfg


# 默认配置
config = PretrainConfig()
config_48g = create_config_48g()
lora_config = LoRAPretrainConfig()
