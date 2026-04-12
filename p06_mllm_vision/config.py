"""
p06 MLLM 多模态视觉微调 - 配置文件

管理多模态大语言模型（MLLM）微调的所有超参数，
包含冻结策略（freeze_vision / partial_unfreeze / full）、
LoRA 配置（应用于 LLM 部分）和图像处理参数。

使用方式:
    from config import config, lora_config, image_config
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. MLLM 微调配置
# ============================================================
@dataclass
class MLLMConfig:
    """多模态大语言模型微调配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct"   # 基座模型
    trust_remote_code: bool = True
    
    # ---- 冻结策略 ----
    # freeze_vision: 冻结视觉编码器，只训练 LLM + 投影层
    # partial_unfreeze: 冻结视觉编码器前 N 层，解冻后几层
    # full: 全模型训练（显存需求大）
    freeze_strategy: str = "freeze_vision"
    vision_unfreeze_layers: int = 4                      # partial_unfreeze 时解冻视觉编码器的最后 N 层
    
    # ---- 数据 ----
    data_dir: str = "data"                               # 数据根目录
    data_file: str = "llava_instruct_20k.json"           # 数据文件名
    max_samples: int = 20000                             # 最大样本数
    val_ratio: float = 0.05                              # 验证集比例
    
    # ---- 训练超参数 ----
    num_train_epochs: int = 3                            # 训练轮数
    per_device_train_batch_size: int = 2                 # 每设备 batch size（VL 模型较大）
    gradient_accumulation_steps: int = 8                 # 梯度累积（等效 batch = 2×8 = 16）
    learning_rate: float = 1e-5                          # 学习率（冻结 vision 时可用较大 lr）
    lr_scheduler_type: str = "cosine"                    # 学习率策略
    warmup_ratio: float = 0.03                           # Warmup 比例
    weight_decay: float = 0.01                           # 权重衰减
    max_grad_norm: float = 1.0                           # 梯度裁剪
    
    # ---- 序列长度 ----
    max_seq_length: int = 2048                           # 最大文本序列长度
    
    # ---- 精度 & 优化 ----
    bf16: bool = True                                    # 使用 bf16 混合精度
    gradient_checkpointing: bool = True                  # 梯度检查点
    
    # ---- 保存 & 日志 ----
    output_dir: str = "outputs/mllm_vision"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 200
    
    # ---- 路径 ----
    seed: int = 42
    log_dir: str = "logs/mllm_vision"


# ============================================================
# 2. LoRA 配置（应用于 LLM 部分）
# ============================================================
@dataclass
class LoRAConfig:
    """LoRA 微调的额外配置（仅作用于 LLM 文本部分）"""
    
    use_lora: bool = True                                # 是否启用 LoRA
    lora_r: int = 16                                     # LoRA 秩（MLLM 微调用中等 rank）
    lora_alpha: int = 32                                 # LoRA alpha（通常 = 2 × r）
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",         # 注意力层
        "gate_proj", "up_proj", "down_proj",             # FFN 层
    ])
    # LoRA 微调的学习率（比全参更大）
    learning_rate: float = 2e-4


# ============================================================
# 3. 图像处理配置
# ============================================================
@dataclass
class ImageConfig:
    """图像预处理参数"""
    
    image_size: int = 448                                # Qwen2.5-VL 默认输入分辨率
    min_pixels: int = 256 * 28 * 28                      # 最小像素数
    max_pixels: int = 1280 * 28 * 28                     # 最大像素数
    
    # 归一化参数（ImageNet 标准）
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # 数据增强
    random_flip: bool = False                            # 微调时通常不翻转
    center_crop: bool = True


# ============================================================
# 4. GPU Preset
# ============================================================
def create_config_24g() -> MLLMConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = MLLMConfig()
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 16                  # 等效 batch = 16
    cfg.max_seq_length = 1024
    cfg.gradient_checkpointing = True
    cfg.freeze_strategy = "freeze_vision"                 # 24G 必须冻结视觉编码器
    return cfg


def create_config_48g() -> MLLMConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = MLLMConfig()
    cfg.per_device_train_batch_size = 2
    cfg.gradient_accumulation_steps = 8
    cfg.max_seq_length = 2048
    cfg.gradient_checkpointing = True
    cfg.freeze_strategy = "partial_unfreeze"              # 48G 可部分解冻视觉编码器
    return cfg


def create_config_80g() -> MLLMConfig:
    """80G GPU (A100) 的优化配置"""
    cfg = MLLMConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4
    cfg.max_seq_length = 2048
    cfg.gradient_checkpointing = False
    cfg.freeze_strategy = "full"                          # 80G 可全参训练
    return cfg


# 默认配置
config = MLLMConfig()
lora_config = LoRAConfig()
image_config = ImageConfig()
