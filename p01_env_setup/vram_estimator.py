"""
p01 环境搭建 - 显存估算工具

根据模型参数量、训练精度、DeepSpeed ZeRO Stage 等参数，
估算训练和推理所需的 GPU 显存，帮助选择合适的配置方案。

核心公式:
    训练显存 = 参数内存 + 梯度内存 + 优化器状态内存 + 激活值内存 + 框架开销

使用方式:
    cd p01_env_setup
    python vram_estimator.py
    python vram_estimator.py --params 0.5  --dtype bf16  --zero 2
    python vram_estimator.py --params 7.0  --dtype bf16  --zero 3  # 估算 7B 模型
"""

import os
import sys
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VRAMEstimatorConfig, vram_config


# ============================================================
# 1. 每参数字节数映射
# ============================================================
BYTES_PER_PARAM = {
    "fp32": 4.0,     # 32-bit 浮点 = 4 字节
    "tf32": 4.0,     # TensorFloat-32（GPU 内部格式，存储仍是 4 字节）
    "bf16": 2.0,     # Brain Float 16 = 2 字节
    "fp16": 2.0,     # IEEE Float 16 = 2 字节
    "int8": 1.0,     # 8-bit 整数 = 1 字节
    "int4": 0.5,     # 4-bit 整数 = 0.5 字节
}


# ============================================================
# 2. 显存估算核心函数
# ============================================================
def estimate_model_memory(
    num_params_billion: float,
    dtype: str = "bf16",
) -> float:
    """
    估算模型参数本身占用的显存。
    
    计算公式:
        参数显存(GB) = 参数量 × 每参数字节数 / 2^30
    
    示例:
        0.5B 模型 + bf16 = 0.5e9 × 2 / 1024^3 ≈ 0.93 GB
        7B 模型 + bf16 = 7e9 × 2 / 1024^3 ≈ 13.04 GB
    
    Args:
        num_params_billion: 参数量（单位：十亿/B）
        dtype: 参数精度
    
    Returns:
        显存占用（GB）
    """
    bytes_per_param = BYTES_PER_PARAM.get(dtype, 2.0)
    return num_params_billion * 1e9 * bytes_per_param / (1024 ** 3)


def estimate_gradient_memory(
    num_params_billion: float,
    dtype: str = "bf16",
) -> float:
    """
    估算梯度占用的显存。
    
    为什么梯度占的显存和参数一样大？
    - 每个参数都有一个对应的梯度值
    - 梯度的精度通常和训练精度相同
    - 混合精度训练中，梯度通常用 fp32 存储以保证精度
    
    注意：混合精度训练时梯度实际是 fp32（4 字节/参数），
    但这里按训练精度估算，实际可能更大。
    """
    # 混合精度训练中，梯度通常用 fp32 累积
    if dtype in ("bf16", "fp16"):
        # 梯度用 fp32 存储
        grad_bytes = 4.0
    else:
        grad_bytes = BYTES_PER_PARAM.get(dtype, 4.0)
    
    return num_params_billion * 1e9 * grad_bytes / (1024 ** 3)


def estimate_optimizer_memory(
    num_params_billion: float,
    optimizer: str = "adamw",
) -> float:
    """
    估算优化器状态占用的显存。
    
    不同优化器的状态量差异很大：
    
    AdamW（最常用）:
    - m (一阶动量): fp32 × 参数量 = 4 字节/参数
    - v (二阶动量): fp32 × 参数量 = 4 字节/参数
    - 总计: 8 字节/参数（是参数本身 bf16 大小的 4 倍！）
    - 这就是为什么优化器状态通常是最大的显存开销
    
    SGD with momentum:
    - m (动量): fp32 × 参数量 = 4 字节/参数
    - 总计: 4 字节/参数
    
    8-bit AdamW (bitsandbytes):
    - m: int8 × 参数量 = 1 字节/参数
    - v: int8 × 参数量 = 1 字节/参数
    - 总计: 2 字节/参数（大幅节省！）
    
    Args:
        num_params_billion: 参数量
        optimizer: "adamw" | "sgd" | "adam_8bit"
    """
    bytes_per_param = {
        "adamw": 8.0,       # m(fp32) + v(fp32) = 4+4
        "sgd": 4.0,         # m(fp32) = 4
        "adam_8bit": 2.0,    # m(int8) + v(int8) = 1+1
    }.get(optimizer, 8.0)
    
    return num_params_billion * 1e9 * bytes_per_param / (1024 ** 3)


def estimate_activation_memory(
    num_params_billion: float,
    batch_size: int = 4,
    seq_length: int = 512,
    num_layers: int = 24,
    hidden_size: int = 896,
    gradient_checkpointing: bool = True,
    dtype: str = "bf16",
) -> float:
    """
    估算激活值占用的显存。
    
    激活值是前向传播过程中每一层的中间结果，
    需要保存下来用于反向传播计算梯度。
    
    无 Gradient Checkpointing:
    - 每层保存完整激活值
    - 显存 ∝ batch_size × seq_len × hidden_size × num_layers
    - 显存消耗最大
    
    有 Gradient Checkpointing:
    - 只保存部分层的激活值（通常是每隔几层保存一次）
    - 反向传播时重新计算被丢弃的激活值
    - 显存降为约 √(num_layers) 倍
    - 代价是训练速度慢约 30%
    
    近似估算公式（来自 Megatron-LM 论文）:
    - 无 checkpointing: 34 * B * S * H * L（bf16）
    - 有 checkpointing: 约 2 * B * S * H * √L（bf16）
    """
    bytes_per_elem = BYTES_PER_PARAM.get(dtype, 2.0)
    
    if gradient_checkpointing:
        # 有 checkpointing: 激活值显存大幅降低
        # 近似公式: 2 * B * S * H * sqrt(L)
        activation_bytes = (
            2.0 * batch_size * seq_length * hidden_size 
            * math.sqrt(num_layers) * bytes_per_elem
        )
    else:
        # 无 checkpointing: 保存每层的完整激活值
        # 近似公式: 34 * B * S * H * L (对于 Transformer)
        # 这里用一个更保守的近似
        activation_bytes = (
            34.0 * batch_size * seq_length * hidden_size 
            * num_layers * bytes_per_elem
        )
    
    return activation_bytes / (1024 ** 3)


def estimate_zero_memory(
    param_memory: float,
    gradient_memory: float,
    optimizer_memory: float,
    zero_stage: int = 0,
    num_gpus: int = 1,
) -> dict:
    """
    根据 DeepSpeed ZeRO Stage 调整显存估算。
    
    ZeRO (Zero Redundancy Optimizer) 的核心思想：
    将优化器状态/梯度/参数分片存储到多张 GPU 上，减少每张卡的显存占用。
    
    单卡场景下的效果：
    
    ZeRO Stage 0: 无优化，所有数据都在同一张卡上
    - 参数: 全量
    - 梯度: 全量
    - 优化器: 全量
    
    ZeRO Stage 1: 优化器状态分片
    - 参数: 全量
    - 梯度: 全量
    - 优化器: 全量 / num_gpus
    - 单卡时等于 Stage 0
    
    ZeRO Stage 2: 优化器 + 梯度分片
    - 参数: 全量
    - 梯度: 全量 / num_gpus
    - 优化器: 全量 / num_gpus
    - 单卡时梯度和优化器不分片，但内部优化内存管理
    
    ZeRO Stage 3: 全部分片（最激进）
    - 参数: 全量 / num_gpus
    - 梯度: 全量 / num_gpus
    - 优化器: 全量 / num_gpus
    - 单卡时开启 CPU offload 可进一步节省
    
    Args:
        param_memory: 参数显存
        gradient_memory: 梯度显存
        optimizer_memory: 优化器显存
        zero_stage: ZeRO 阶段 (0/1/2/3)
        num_gpus: GPU 数量
    
    Returns:
        dict 包含各项显存分配
    """
    n = max(num_gpus, 1)
    
    if zero_stage == 0:
        return {
            "params": param_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
        }
    elif zero_stage == 1:
        return {
            "params": param_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory / n,
        }
    elif zero_stage == 2:
        return {
            "params": param_memory,
            "gradients": gradient_memory / n,
            "optimizer": optimizer_memory / n,
        }
    elif zero_stage == 3:
        return {
            "params": param_memory / n,
            "gradients": gradient_memory / n,
            "optimizer": optimizer_memory / n,
        }
    else:
        raise ValueError(f"Unknown ZeRO stage: {zero_stage}")


# ============================================================
# 3. 综合估算函数
# ============================================================
def estimate_training_vram(
    num_params_billion: float,
    dtype: str = "bf16",
    zero_stage: int = 2,
    num_gpus: int = 1,
    batch_size: int = 4,
    seq_length: int = 512,
    gradient_checkpointing: bool = True,
    optimizer: str = "adamw",
) -> dict:
    """
    综合估算训练所需的显存。
    
    总显存 = (参数 + 梯度 + 优化器) × ZeRO 调整 + 激活值 + 框架开销
    """
    # 根据参数量推断模型架构参数
    # Qwen2.5 系列的近似映射
    model_configs = {
        0.5: {"layers": 24, "hidden": 896},
        1.5: {"layers": 28, "hidden": 1536},
        3.0: {"layers": 36, "hidden": 2048},
        7.0: {"layers": 28, "hidden": 3584},
    }
    
    # 找最近的配置
    closest = min(model_configs.keys(), key=lambda x: abs(x - num_params_billion))
    model_cfg = model_configs[closest]
    
    # 各项显存
    param_mem = estimate_model_memory(num_params_billion, dtype)
    grad_mem = estimate_gradient_memory(num_params_billion, dtype)
    optim_mem = estimate_optimizer_memory(num_params_billion, optimizer)
    
    # ZeRO 调整
    zero_result = estimate_zero_memory(
        param_mem, grad_mem, optim_mem, zero_stage, num_gpus
    )
    
    # 激活值
    act_mem = estimate_activation_memory(
        num_params_billion,
        batch_size=batch_size,
        seq_length=seq_length,
        num_layers=model_cfg["layers"],
        hidden_size=model_cfg["hidden"],
        gradient_checkpointing=gradient_checkpointing,
        dtype=dtype,
    )
    
    # 框架开销（CUDA context + 临时 buffer，经验值约 1-2 GB）
    overhead = 1.5
    
    total = sum(zero_result.values()) + act_mem + overhead
    
    return {
        "params_gb": zero_result["params"],
        "gradients_gb": zero_result["gradients"],
        "optimizer_gb": zero_result["optimizer"],
        "activations_gb": act_mem,
        "overhead_gb": overhead,
        "total_gb": total,
        "config": {
            "model_params": f"{num_params_billion}B",
            "dtype": dtype,
            "zero_stage": zero_stage,
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "gradient_checkpointing": gradient_checkpointing,
            "optimizer": optimizer,
        },
    }


def estimate_inference_vram(
    num_params_billion: float,
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_length: int = 2048,
) -> dict:
    """
    估算推理所需的显存。
    
    推理显存 = 模型参数 + KV Cache + 框架开销
    
    推理比训练省很多显存，因为：
    1. 不需要梯度
    2. 不需要优化器状态
    3. 激活值不需要保存（没有反向传播）
    4. 只有 KV Cache 额外开销
    """
    model_configs = {
        0.5: {"layers": 24, "hidden": 896, "kv_heads": 2},
        1.5: {"layers": 28, "hidden": 1536, "kv_heads": 2},
        3.0: {"layers": 36, "hidden": 2048, "kv_heads": 2},
        7.0: {"layers": 28, "hidden": 3584, "kv_heads": 4},
    }
    
    closest = min(model_configs.keys(), key=lambda x: abs(x - num_params_billion))
    model_cfg = model_configs[closest]
    
    # 模型参数显存
    param_mem = estimate_model_memory(num_params_billion, dtype)
    
    # KV Cache 显存
    # KV Cache 大小 = 2 (K+V) × layers × kv_heads × head_dim × seq_len × batch × bytes
    head_dim = model_cfg["hidden"] // (model_cfg["kv_heads"] * 4)  # 近似
    kv_bytes = (
        2 * model_cfg["layers"] * model_cfg["kv_heads"] * head_dim
        * seq_length * batch_size * BYTES_PER_PARAM.get(dtype, 2.0)
    )
    kv_mem = kv_bytes / (1024 ** 3)
    
    overhead = 1.0
    total = param_mem + kv_mem + overhead
    
    return {
        "params_gb": param_mem,
        "kv_cache_gb": kv_mem,
        "overhead_gb": overhead,
        "total_gb": total,
    }


# ============================================================
# 4. 对比表格输出
# ============================================================
def print_comparison_table(num_params_billion: float = 0.5):
    """打印多精度 × 多 ZeRO Stage 的显存对比表"""
    
    print(f"\n{'='*80}")
    print(f"  {num_params_billion}B 模型训练显存估算对比")
    print(f"  (batch_size=4, seq_length=512, gradient_checkpointing=True, AdamW)")
    print(f"{'='*80}")
    
    dtypes = ["fp32", "bf16", "fp16", "int8", "int4"]
    
    # 精度对比（ZeRO-2）
    print(f"\n📊 不同精度下的显存占用（ZeRO Stage 2, 单卡）")
    print(f"{'精度':<8} {'参数':>8} {'梯度':>8} {'优化器':>8} {'激活值':>8} {'总计':>8}")
    print("-" * 56)
    
    for dtype in dtypes:
        result = estimate_training_vram(
            num_params_billion, dtype=dtype, zero_stage=2,
            batch_size=4, seq_length=512,
        )
        print(
            f"{dtype:<8} "
            f"{result['params_gb']:>7.2f}G "
            f"{result['gradients_gb']:>7.2f}G "
            f"{result['optimizer_gb']:>7.2f}G "
            f"{result['activations_gb']:>7.2f}G "
            f"{result['total_gb']:>7.2f}G"
        )
    
    # ZeRO Stage 对比（bf16）
    print(f"\n📊 不同 ZeRO Stage 的显存占用（bf16, 单卡）")
    print(f"{'ZeRO':<10} {'参数':>8} {'梯度':>8} {'优化器':>8} {'激活值':>8} {'总计':>8}")
    print("-" * 58)
    
    for stage in [0, 1, 2, 3]:
        result = estimate_training_vram(
            num_params_billion, dtype="bf16", zero_stage=stage,
            batch_size=4, seq_length=512,
        )
        print(
            f"Stage {stage:<4} "
            f"{result['params_gb']:>7.2f}G "
            f"{result['gradients_gb']:>7.2f}G "
            f"{result['optimizer_gb']:>7.2f}G "
            f"{result['activations_gb']:>7.2f}G "
            f"{result['total_gb']:>7.2f}G"
        )
    
    # Gradient Checkpointing 对比
    print(f"\n📊 Gradient Checkpointing 开/关对比（bf16, ZeRO-2, 单卡）")
    print(f"{'GC':<8} {'激活值':>8} {'总计':>8} {'节省':>8}")
    print("-" * 36)
    
    for gc in [False, True]:
        result = estimate_training_vram(
            num_params_billion, dtype="bf16", zero_stage=2,
            batch_size=4, seq_length=512,
            gradient_checkpointing=gc,
        )
        gc_str = "开启" if gc else "关闭"
        save_str = "" if not gc else ""
        print(
            f"{gc_str:<8} "
            f"{result['activations_gb']:>7.2f}G "
            f"{result['total_gb']:>7.2f}G "
            f"{save_str:>8}"
        )
    
    result_off = estimate_training_vram(
        num_params_billion, dtype="bf16", zero_stage=2,
        batch_size=4, seq_length=512, gradient_checkpointing=False,
    )
    result_on = estimate_training_vram(
        num_params_billion, dtype="bf16", zero_stage=2,
        batch_size=4, seq_length=512, gradient_checkpointing=True,
    )
    save = result_off["total_gb"] - result_on["total_gb"]
    print(f"\n  → Gradient Checkpointing 节省约 {save:.2f} GB 显存")


def print_multi_model_table():
    """打印多模型大小的显存估算对比"""
    
    print(f"\n{'='*80}")
    print(f"  多模型大小显存估算对比")
    print(f"  (bf16, ZeRO-2, batch_size=4, seq_length=512, GC=True)")
    print(f"{'='*80}")
    
    model_sizes = [0.5, 1.5, 3.0, 7.0]
    
    print(f"\n{'模型':<8} {'训练总计':>10} {'推理总计':>10} {'24G可训练':>10} {'48G可训练':>10}")
    print("-" * 52)
    
    for size in model_sizes:
        train = estimate_training_vram(size, "bf16", zero_stage=2, batch_size=4, seq_length=512)
        infer = estimate_inference_vram(size, "bf16")
        
        can_24g = "✅" if train["total_gb"] < 24 else "❌"
        can_48g = "✅" if train["total_gb"] < 48 else "❌"
        
        print(
            f"{size}B{'':<4} "
            f"{train['total_gb']:>9.2f}G "
            f"{infer['total_gb']:>9.2f}G "
            f"{'':>4}{can_24g:>6} "
            f"{'':>4}{can_48g:>6}"
        )


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GPU 显存估算工具")
    parser.add_argument("--params", type=float, default=0.5,
                       help="模型参数量（B）, 默认 0.5")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp32", "bf16", "fp16", "int8", "int4"])
    parser.add_argument("--zero", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--no-gc", action="store_true", help="禁用 gradient checkpointing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  实践大模型 - 显存估算工具")
    print("=" * 60)
    
    # 单模型详细估算
    result = estimate_training_vram(
        num_params_billion=args.params,
        dtype=args.dtype,
        zero_stage=args.zero,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        gradient_checkpointing=not args.no_gc,
    )
    
    print(f"\n📋 训练显存估算结果")
    print("-" * 40)
    print(f"  模型:                {args.params}B")
    print(f"  精度:                {args.dtype}")
    print(f"  ZeRO Stage:          {args.zero}")
    print(f"  Batch Size:          {args.batch_size}")
    print(f"  Seq Length:          {args.seq_length}")
    print(f"  Gradient Checkpoint: {'开启' if not args.no_gc else '关闭'}")
    print()
    print(f"  参数显存:            {result['params_gb']:.2f} GB")
    print(f"  梯度显存:            {result['gradients_gb']:.2f} GB")
    print(f"  优化器显存:          {result['optimizer_gb']:.2f} GB")
    print(f"  激活值显存:          {result['activations_gb']:.2f} GB")
    print(f"  框架开销:            {result['overhead_gb']:.2f} GB")
    print(f"  ─────────────────────────")
    print(f"  📊 总计:             {result['total_gb']:.2f} GB")
    
    # 检查是否适合当前 GPU
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            if result["total_gb"] < vram * 0.9:  # 留 10% 余量
                print(f"\n  ✅ 当前 GPU ({vram:.0f}GB) 可以训练！")
            else:
                print(f"\n  ❌ 当前 GPU ({vram:.0f}GB) 显存不足")
                print(f"     建议: 降低 batch_size / 开启 gradient checkpointing / 升级 ZeRO Stage")
    except ImportError:
        pass
    
    # 对比表格
    print_comparison_table(args.params)
    print_multi_model_table()
    
    # 推理估算
    infer = estimate_inference_vram(args.params, args.dtype)
    print(f"\n📋 推理显存估算")
    print("-" * 40)
    print(f"  模型参数:  {infer['params_gb']:.2f} GB")
    print(f"  KV Cache:  {infer['kv_cache_gb']:.2f} GB")
    print(f"  框架开销:  {infer['overhead_gb']:.2f} GB")
    print(f"  📊 总计:   {infer['total_gb']:.2f} GB")


if __name__ == "__main__":
    main()
