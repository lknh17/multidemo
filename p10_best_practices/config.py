"""
p10 最佳实践总结 - 配置文件
"""
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class BestPracticesConfig:
    """各阶段推荐配置汇总"""
    
    # ---- 推荐配置速查表 ----
    pretrain_config: dict = field(default_factory=lambda: {
        "model": "Qwen/Qwen2.5-0.5B", "dtype": "bf16", "lr": 2e-5,
        "scheduler": "cosine", "warmup": 0.05, "batch": 16,
        "zero_stage": 0, "gc": True, "epochs": 1,
    })
    sft_config: dict = field(default_factory=lambda: {
        "method": "lora", "r": 32, "alpha": 64, "lr": 2e-4,
        "epochs": 3, "batch": 16,
    })
    dpo_config: dict = field(default_factory=lambda: {
        "algorithm": "dpo", "beta": 0.1, "lr": 5e-5,
        "epochs": 1, "batch": 8,
    })
    rl_config: dict = field(default_factory=lambda: {
        "algorithm": "grpo", "group_size": 8, "temperature": 0.7,
        "kl_coef": 0.05, "lr": 1e-5,
    })
    
    # ---- 路径 ----
    model_stages: dict = field(default_factory=lambda: {
        "base": "Qwen/Qwen2.5-0.5B",
        "pretrained": "../p02_continual_pretrain/outputs/pretrain/final",
        "sft": "../p03_sft_finetuning/outputs/sft/final",
        "dpo": "../p04_dpo_alignment/outputs/dpo/final",
        "rl": "../p05_rl_grpo/outputs/grpo/final",
    })

config = BestPracticesConfig()
