"""
p01 环境搭建 - GPU 性能基准测试

测试 GPU 的核心性能指标：
1. 矩阵乘法吞吐量（TFLOPS）—— 衡量计算能力
2. 显存带宽（GB/s）—— 衡量数据传输速度
3. 不同精度的性能差异（fp32 vs fp16 vs bf16）

为什么需要 Benchmark？
- 了解你的 GPU 的实际性能瓶颈（计算密集 or 内存密集）
- 对比不同精度的加速比，选择最优训练精度
- 检测 GPU 是否存在性能异常（过热降频、驱动问题等）

使用方式:
    cd p01_env_setup
    python gpu_benchmark.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 矩阵乘法吞吐测试
# ============================================================
def benchmark_matmul(
    M: int, N: int, K: int,
    dtype_str: str = "float16",
    warmup_iters: int = 10,
    test_iters: int = 100,
    device: str = "cuda",
) -> dict:
    """
    测试矩阵乘法的吞吐量。
    
    矩阵乘法（GEMM）是大模型训练中最核心的计算操作：
    - 注意力计算: Q × K^T，Attn × V
    - FFN 前向: x × W1，x × W2
    - 几乎所有计算都是矩阵乘法
    
    测量指标: TFLOPS (Tera Floating-point Operations Per Second)
    FLOPS 计算: 2 × M × N × K（乘法+加法）
    
    Args:
        M, N, K: 矩阵维度 (A: M×K, B: K×N, C: M×N)
        dtype_str: 数据类型
        warmup_iters: 预热次数（让 GPU 达到稳定频率）
        test_iters: 测试次数
    
    Returns:
        包含 TFLOPS、延迟等信息的字典
    """
    import torch
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    
    # 创建随机矩阵
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # 预热（GPU 需要"暖机"才能达到最高频率）
    for _ in range(warmup_iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # 正式测试
    start = time.time()
    for _ in range(test_iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算 TFLOPS
    # 一次矩阵乘法的浮点运算量: 2 * M * N * K（每个输出元素需要 K 次乘法 + K 次加法）
    flops_per_op = 2.0 * M * N * K
    total_flops = flops_per_op * test_iters
    tflops = total_flops / elapsed / 1e12
    
    avg_latency_ms = elapsed / test_iters * 1000
    
    return {
        "shape": f"({M}×{K}) × ({K}×{N})",
        "dtype": dtype_str,
        "tflops": tflops,
        "latency_ms": avg_latency_ms,
        "total_time_s": elapsed,
    }


# ============================================================
# 2. 显存带宽测试
# ============================================================
def benchmark_bandwidth(
    size_gb: float = 1.0,
    test_iters: int = 50,
    device: str = "cuda",
) -> dict:
    """
    测试 GPU 显存带宽。
    
    大模型训练经常是"内存受限"(memory-bound)而非"计算受限"：
    - 训练时需要频繁从显存读写大量参数和激活值
    - 推理时逐 token 生成，每步都要读取全部 KV Cache
    - 带宽不够会导致 GPU 计算单元空闲等待数据
    
    Args:
        size_gb: 测试数据大小(GB)
        test_iters: 测试次数
    
    Returns:
        包含带宽等信息的字典
    """
    import torch
    
    num_elements = int(size_gb * 1024 ** 3 / 4)  # float32 = 4 bytes
    
    src = torch.randn(num_elements, device=device)
    dst = torch.empty_like(src)
    
    # 预热
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    for _ in range(test_iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 带宽 = 数据量 / 时间
    # 读 + 写 = 2x 数据量
    bytes_transferred = 2 * num_elements * 4 * test_iters  # 读+写
    bandwidth_gbps = bytes_transferred / elapsed / (1024 ** 3)
    
    return {
        "size_gb": size_gb,
        "bandwidth_gbps": bandwidth_gbps,
        "latency_ms": elapsed / test_iters * 1000,
    }


# ============================================================
# 3. Flash Attention 性能测试
# ============================================================
def benchmark_attention(
    batch_size: int = 4,
    seq_length: int = 2048,
    num_heads: int = 16,
    head_dim: int = 64,
    device: str = "cuda",
) -> dict:
    """测试标准 Attention vs Flash Attention 的性能差异"""
    import torch
    import torch.nn.functional as F
    
    # 创建 QKV
    q = torch.randn(batch_size, num_heads, seq_length, head_dim, 
                     device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    results = {}
    
    # 标准 Attention
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / 10 * 1000
    results["standard_ms"] = std_time
    
    # Flash Attention（如果可用）
    try:
        from torch.nn.functional import scaled_dot_product_attention
        
        # 使用 PyTorch 原生 SDPA（内部会自动选择 Flash Attention）
        q_sdpa = q.transpose(1, 2).contiguous()  # [B, S, H, D]
        k_sdpa = k.transpose(1, 2).contiguous()
        v_sdpa = v.transpose(1, 2).contiguous()
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            out = scaled_dot_product_attention(
                q_sdpa.transpose(1, 2), k_sdpa.transpose(1, 2), v_sdpa.transpose(1, 2),
                is_causal=True,
            )
        torch.cuda.synchronize()
        flash_time = (time.time() - start) / 10 * 1000
        results["flash_ms"] = flash_time
        results["speedup"] = std_time / flash_time
    except Exception as e:
        results["flash_ms"] = None
        results["speedup"] = None
        results["flash_error"] = str(e)
    
    return results


# ============================================================
# 4. 主流程
# ============================================================
def main():
    import torch
    
    print("=" * 60)
    print("  实践大模型 - GPU 性能基准测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("\n❌ 未检测到 CUDA GPU，无法运行 benchmark")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n🖥️  GPU: {gpu_name} | 显存: {vram:.1f} GB")
    
    # ---- 矩阵乘法吞吐测试 ----
    print(f"\n{'='*60}")
    print("  📊 矩阵乘法吞吐量测试 (TFLOPS)")
    print(f"{'='*60}")
    
    shapes = [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    dtypes = ["float32", "float16", "bfloat16"]
    
    print(f"\n{'Shape':<25} {'fp32':>10} {'fp16':>10} {'bf16':>10}")
    print("-" * 58)
    
    for M, N, K in shapes:
        row = f"({M}×{K})×({K}×{N})"
        values = []
        for dtype_str in dtypes:
            try:
                result = benchmark_matmul(M, N, K, dtype_str=dtype_str)
                values.append(f"{result['tflops']:.1f}")
            except Exception:
                values.append("N/A")
        print(f"{row:<25} {values[0]:>10} {values[1]:>10} {values[2]:>10}")
    
    # fp16 vs fp32 加速比
    try:
        fp32 = benchmark_matmul(4096, 4096, 4096, "float32")
        fp16 = benchmark_matmul(4096, 4096, 4096, "float16")
        bf16 = benchmark_matmul(4096, 4096, 4096, "bfloat16")
        print(f"\n  fp16/fp32 加速比: {fp16['tflops']/fp32['tflops']:.2f}x")
        print(f"  bf16/fp32 加速比: {bf16['tflops']/fp32['tflops']:.2f}x")
    except Exception:
        pass
    
    # ---- 显存带宽测试 ----
    print(f"\n{'='*60}")
    print("  📊 显存带宽测试 (GB/s)")
    print(f"{'='*60}")
    
    print(f"\n{'数据大小':<15} {'带宽(GB/s)':>12} {'延迟(ms)':>10}")
    print("-" * 40)
    
    for size in [0.1, 0.5, 1.0, 2.0]:
        try:
            result = benchmark_bandwidth(size_gb=size)
            print(f"{size:.1f} GB{'':<10} {result['bandwidth_gbps']:>11.1f} {result['latency_ms']:>9.2f}")
        except Exception:
            print(f"{size:.1f} GB{'':<10} {'N/A':>11} {'N/A':>9}")
    
    # ---- Attention 性能测试 ----
    print(f"\n{'='*60}")
    print("  📊 Attention 性能测试")
    print(f"{'='*60}")
    
    try:
        attn_result = benchmark_attention()
        print(f"\n  标准 Attention:  {attn_result['standard_ms']:.2f} ms")
        if attn_result.get("flash_ms"):
            print(f"  Flash Attention: {attn_result['flash_ms']:.2f} ms")
            print(f"  加速比:          {attn_result['speedup']:.2f}x")
        else:
            print(f"  Flash Attention: 不可用 ({attn_result.get('flash_error', 'unknown')})")
    except Exception as e:
        print(f"\n  ⚠️ Attention 测试失败: {e}")
    
    # ---- 理论性能参考 ----
    print(f"\n{'='*60}")
    print("  📋 GPU 理论性能参考")
    print(f"{'='*60}")
    print(f"""
  GPU 型号         fp32 TFLOPS  fp16 TFLOPS  显存带宽(GB/s)  显存(GB)
  ─────────────────────────────────────────────────────────────────
  RTX 4090         82.6         165.2        1008            24
  RTX A6000        38.7         77.4         768             48
  A100 (SXM)       19.5         312.0        2039            80
  H100 (SXM)       66.9         989.4        3352            80
  
  💡 实测值通常是理论值的 60-80%，受到散热、功耗墙等因素影响。
  💡 大模型训练主要受 显存带宽 而非 计算吞吐 限制（memory-bound）。
""")
    
    print("=" * 60)
    print("  ✅ GPU 性能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
