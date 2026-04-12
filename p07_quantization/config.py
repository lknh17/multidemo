"""
p07 模型量化 - 配置文件

管理四种量化方法（GPTQ/AWQ/GGUF/bitsandbytes）的所有参数，
支持 24G/48G 两套 GPU preset，以及 benchmark 评测参数。

使用方式:
    from config import config, gptq_config, awq_config, gguf_config, bnb_config
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 基础量化配置
# ============================================================
@dataclass
class QuantBaseConfig:
    """量化公共配置"""
    
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B"          # 原始模型（FP16/BF16）
    model_revision: Optional[str] = None              # 模型版本
    trust_remote_code: bool = True                    # Qwen 系列需要
    
    # ---- 校准数据 ----
    calibration_dataset: str = "wikitext"             # 校准数据集
    calibration_split: str = "train"
    calibration_samples: int = 128                    # 校准样本数（128 通常足够）
    calibration_seq_length: int = 512                 # 校准序列长度
    
    # ---- 输出 ----
    output_base_dir: str = "outputs/quantized"        # 量化模型输出根目录
    
    # ---- 评测 ----
    eval_dataset: str = "wikitext"
    eval_split: str = "test"
    eval_samples: int = 256                           # 评测样本数
    
    # ---- 路径 ----
    seed: int = 42
    device: str = "auto"                              # auto / cuda / cpu


# ============================================================
# 2. GPTQ 量化配置
# ============================================================
@dataclass
class GPTQConfig:
    """GPTQ 量化参数"""
    
    bits: int = 4                                     # 量化位数：2/3/4/8
    group_size: int = 128                             # 分组量化大小（128 是最佳平衡）
    desc_act: bool = True                             # 按激活值降序排列（更精确但更慢）
    sym: bool = True                                  # 对称量化（推荐 True）
    damp_percent: float = 0.01                        # Hessian 阻尼百分比
    use_exllama: bool = True                          # 使用 ExLlama 内核加速推理
    
    # ---- 输出 ----
    output_dir: str = "outputs/quantized/gptq-4bit"


# ============================================================
# 3. AWQ 量化配置
# ============================================================
@dataclass
class AWQConfig:
    """AWQ (Activation-aware Weight Quantization) 量化参数"""
    
    bits: int = 4                                     # 量化位数：4
    group_size: int = 128                             # 分组量化大小
    zero_point: bool = True                           # 是否使用零点（非对称量化）
    version: str = "GEMM"                             # 内核版本: GEMM / GEMV
    
    # ---- 输出 ----
    output_dir: str = "outputs/quantized/awq-4bit"


# ============================================================
# 4. GGUF 量化配置
# ============================================================
@dataclass
class GGUFConfig:
    """GGUF (llama.cpp) 量化参数"""
    
    # 支持的量化级别（从最小到最大）
    quant_types: List[str] = field(default_factory=lambda: [
        "Q2_K",     # 2-bit（极致压缩，质量损失大）
        "Q3_K_M",   # 3-bit medium
        "Q4_K_M",   # 4-bit medium（推荐，平衡质量和大小）
        "Q5_K_M",   # 5-bit medium
        "Q6_K",     # 6-bit（接近原始质量）
        "Q8_0",     # 8-bit（几乎无损）
    ])
    default_quant_type: str = "Q4_K_M"               # 默认量化级别
    
    # ---- llama.cpp 路径 ----
    llama_cpp_path: str = "llama.cpp"                 # llama.cpp 仓库路径
    
    # ---- 输出 ----
    output_dir: str = "outputs/quantized/gguf"


# ============================================================
# 5. bitsandbytes 量化配置
# ============================================================
@dataclass
class BnBConfig:
    """bitsandbytes 量化参数"""
    
    load_in_4bit: bool = True                         # 4-bit 量化
    load_in_8bit: bool = False                        # 8-bit 量化（与 4bit 互斥）
    bnb_4bit_quant_type: str = "nf4"                  # nf4 / fp4
    bnb_4bit_compute_dtype: str = "bfloat16"          # 计算精度
    bnb_4bit_use_double_quant: bool = True            # 双重量化（进一步压缩）
    
    # ---- 输出 ----
    output_dir: str = "outputs/quantized/bnb-nf4"


# ============================================================
# 6. Benchmark 配置
# ============================================================
@dataclass
class BenchmarkConfig:
    """量化方法对比评测配置"""
    
    # ---- 评测任务 ----
    eval_perplexity: bool = True                      # 计算困惑度
    eval_inference_speed: bool = True                  # 测量推理速度
    eval_model_size: bool = True                      # 模型大小
    eval_vram_usage: bool = True                      # 显存占用
    
    # ---- 推理测试 ----
    warmup_runs: int = 3                              # 预热轮数
    benchmark_runs: int = 10                          # 正式测试轮数
    input_length: int = 128                           # 输入长度
    output_length: int = 64                           # 生成长度
    
    # ---- 报告 ----
    report_dir: str = "outputs/quantized/benchmark"


# ============================================================
# 7. GPU Preset
# ============================================================
def create_config_24g() -> QuantBaseConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = QuantBaseConfig()
    cfg.calibration_samples = 128
    cfg.calibration_seq_length = 512
    return cfg


def create_config_48g() -> QuantBaseConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = QuantBaseConfig()
    cfg.calibration_samples = 256              # 更多校准样本
    cfg.calibration_seq_length = 1024          # 更长的校准序列
    return cfg


# 默认配置
config = QuantBaseConfig()
config_48g = create_config_48g()
gptq_config = GPTQConfig()
awq_config = AWQConfig()
gguf_config = GGUFConfig()
bnb_config = BnBConfig()
benchmark_config = BenchmarkConfig()
