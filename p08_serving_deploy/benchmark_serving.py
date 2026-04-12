"""
p08 推理部署 - 推理服务压力测试

在不同并发级别下测试推理服务性能，测量：
- TTFT (Time To First Token): 首 token 延迟
- TPS (Tokens Per Second): 每秒生成 token 数
- Throughput: 吞吐量（请求/秒）
- P50/P90/P99 延迟

生成性能对比图表。

使用方式:
    python benchmark_serving.py
    python benchmark_serving.py --url http://localhost:8001/v1 --concurrency 1,4,8,16
"""

import os
import sys
import time
import json
import asyncio
import argparse
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config, benchmark_config

try:
    import aiohttp
except ImportError:
    print("请安装 aiohttp: pip install aiohttp")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# 1. 请求结果数据结构
# ============================================================
@dataclass
class RequestResult:
    """单次请求的结果"""
    prompt: str = ""
    ttft: float = 0.0                 # 首 token 延迟（秒）
    total_time: float = 0.0           # 总耗时（秒）
    output_tokens: int = 0            # 生成 token 数
    tps: float = 0.0                  # Tokens/second
    success: bool = True
    error: Optional[str] = None


@dataclass
class ConcurrencyResult:
    """某并发级别的汇总结果"""
    concurrency: int = 0
    num_requests: int = 0
    num_success: int = 0
    num_failed: int = 0
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0
    avg_tps: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0           # 请求/秒
    total_tokens: int = 0
    total_time: float = 0.0


# ============================================================
# 2. 异步请求发送
# ============================================================
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    """发送单次流式请求并测量性能"""
    result = RequestResult(prompt=prompt)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0

    try:
        async with session.post(
            f"{url}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                result.success = False
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except asyncio.TimeoutError:
        result.success = False
        result.error = "请求超时"
        return result
    except Exception as e:
        result.success = False
        result.error = str(e)
        return result

    end_time = time.perf_counter()
    result.total_time = end_time - start_time
    result.output_tokens = output_tokens

    if first_token_time is not None:
        result.ttft = first_token_time - start_time
    if output_tokens > 0 and result.total_time > 0:
        result.tps = output_tokens / result.total_time

    return result


# ============================================================
# 3. 并发测试执行器
# ============================================================
async def run_concurrency_test(
    url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
) -> ConcurrencyResult:
    """执行某并发级别的压力测试"""
    print(f"\n  并发={concurrency}: 发送 {num_requests} 个请求...")

    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def bounded_request(prompt):
        async with semaphore:
            return await send_request(
                session, url, model, prompt, max_tokens, temperature
            )

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 准备请求 prompt
        request_prompts = [random.choice(prompts) for _ in range(num_requests)]

        start = time.perf_counter()
        tasks = [bounded_request(p) for p in request_prompts]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    # 汇总结果
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    cr = ConcurrencyResult(
        concurrency=concurrency,
        num_requests=num_requests,
        num_success=len(success),
        num_failed=len(failed),
        total_time=total_time,
    )

    if success:
        ttfts = [r.ttft for r in success if r.ttft > 0]
        latencies = [r.total_time for r in success]
        tps_list = [r.tps for r in success if r.tps > 0]

        if ttfts:
            cr.avg_ttft = statistics.mean(ttfts)
            cr.p50_ttft = sorted(ttfts)[len(ttfts) // 2]
            cr.p90_ttft = sorted(ttfts)[int(len(ttfts) * 0.9)]
            cr.p99_ttft = sorted(ttfts)[min(int(len(ttfts) * 0.99), len(ttfts) - 1)]

        if latencies:
            cr.avg_latency = statistics.mean(latencies)
            cr.p50_latency = sorted(latencies)[len(latencies) // 2]
            cr.p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
            cr.p99_latency = sorted(latencies)[min(int(len(latencies) * 0.99), len(latencies) - 1)]

        if tps_list:
            cr.avg_tps = statistics.mean(tps_list)

        cr.total_tokens = sum(r.output_tokens for r in success)
        cr.throughput = len(success) / total_time

    # 打印结果
    print(f"    成功: {cr.num_success}, 失败: {cr.num_failed}")
    print(f"    TTFT   avg={cr.avg_ttft:.3f}s  p50={cr.p50_ttft:.3f}s  p90={cr.p90_ttft:.3f}s  p99={cr.p99_ttft:.3f}s")
    print(f"    延迟   avg={cr.avg_latency:.3f}s  p50={cr.p50_latency:.3f}s  p90={cr.p90_latency:.3f}s")
    print(f"    TPS={cr.avg_tps:.1f}  吞吐={cr.throughput:.2f} req/s  总token={cr.total_tokens}")

    return cr


# ============================================================
# 4. 生成对比图表
# ============================================================
def generate_charts(results: List[ConcurrencyResult], output_dir: str):
    """生成性能对比图表"""
    if not HAS_MATPLOTLIB:
        print("\n  ⚠ matplotlib 未安装，跳过图表生成。")
        return

    os.makedirs(output_dir, exist_ok=True)
    concurrencies = [r.concurrency for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("推理服务压力测试结果", fontsize=16, fontweight="bold")

    # 1. TTFT
    ax = axes[0][0]
    ax.plot(concurrencies, [r.avg_ttft for r in results], "o-", label="avg", color="#0984E3")
    ax.plot(concurrencies, [r.p50_ttft for r in results], "s--", label="p50", color="#00B894")
    ax.plot(concurrencies, [r.p90_ttft for r in results], "^--", label="p90", color="#F59E0B")
    ax.plot(concurrencies, [r.p99_ttft for r in results], "D--", label="p99", color="#EF4444")
    ax.set_xlabel("并发数")
    ax.set_ylabel("TTFT (秒)")
    ax.set_title("首 Token 延迟 (TTFT)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. TPS
    ax = axes[0][1]
    ax.bar(concurrencies, [r.avg_tps for r in results], color="#6C5CE7", alpha=0.8)
    ax.set_xlabel("并发数")
    ax.set_ylabel("TPS (tokens/s)")
    ax.set_title("每秒生成 Token 数 (TPS)")
    ax.grid(True, alpha=0.3)

    # 3. 吞吐量
    ax = axes[1][0]
    ax.plot(concurrencies, [r.throughput for r in results], "o-", color="#00B894", linewidth=2)
    ax.fill_between(concurrencies, [r.throughput for r in results], alpha=0.2, color="#00B894")
    ax.set_xlabel("并发数")
    ax.set_ylabel("吞吐量 (req/s)")
    ax.set_title("请求吞吐量")
    ax.grid(True, alpha=0.3)

    # 4. 延迟分布
    ax = axes[1][1]
    ax.plot(concurrencies, [r.avg_latency for r in results], "o-", label="avg", color="#0984E3")
    ax.plot(concurrencies, [r.p50_latency for r in results], "s--", label="p50", color="#00B894")
    ax.plot(concurrencies, [r.p90_latency for r in results], "^--", label="p90", color="#F59E0B")
    ax.plot(concurrencies, [r.p99_latency for r in results], "D--", label="p99", color="#EF4444")
    ax.set_xlabel("并发数")
    ax.set_ylabel("延迟 (秒)")
    ax.set_title("端到端延迟")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 图表已保存: {chart_path}")


# ============================================================
# 5. 主流程
# ============================================================
async def run_benchmark(args):
    """运行完整压力测试"""
    url = args.url or benchmark_config.target_url
    model = args.model or benchmark_config.model_name

    # 解析并发级别
    if args.concurrency:
        levels = [int(x) for x in args.concurrency.split(",")]
    else:
        levels = benchmark_config.concurrency_levels

    num_req = args.num_requests or benchmark_config.num_requests_per_level
    max_tokens = args.max_tokens or benchmark_config.max_tokens
    prompts = benchmark_config.test_prompts

    print("=" * 60)
    print("  推理服务压力测试")
    print("=" * 60)
    print(f"  目标服务:   {url}")
    print(f"  模型:       {model}")
    print(f"  并发级别:   {levels}")
    print(f"  每级请求数: {num_req}")
    print(f"  最大 token: {max_tokens}")
    print("=" * 60)

    # 连通性测试
    print("\n  🔍 检查服务连通性...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    print(f"  ✓ 服务可用，可用模型: {models}")
                else:
                    print(f"  ⚠ 服务返回 {resp.status}，继续尝试...")
    except Exception as e:
        print(f"  ✗ 无法连接服务: {e}")
        print(f"  请确保推理服务已在 {url} 上运行。")
        return

    # 逐级执行测试
    all_results: List[ConcurrencyResult] = []
    for level in levels:
        result = await run_concurrency_test(
            url=url,
            model=model,
            concurrency=level,
            num_requests=num_req,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=benchmark_config.temperature,
        )
        all_results.append(result)

    # 汇总表格
    print("\n" + "=" * 80)
    print(f"  {'并发':>4}  {'成功':>4}  {'TTFT avg':>9}  {'TTFT p90':>9}  {'TPS':>7}  {'吞吐':>8}  {'延迟 avg':>9}  {'延迟 p90':>9}")
    print("-" * 80)
    for r in all_results:
        print(f"  {r.concurrency:>4}  {r.num_success:>4}  {r.avg_ttft:>8.3f}s  {r.p90_ttft:>8.3f}s  {r.avg_tps:>6.1f}  {r.throughput:>7.2f}  {r.avg_latency:>8.3f}s  {r.p90_latency:>8.3f}s")
    print("=" * 80)

    # 保存结果
    output_dir = args.output_dir or benchmark_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
    print(f"\n  💾 结果已保存: {results_path}")

    # 生成图表
    generate_charts(all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description="推理服务压力测试")
    parser.add_argument("--url", type=str, default=None, help="服务地址 (如 http://localhost:8000/v1)")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--concurrency", type=str, default=None, help="并发级别，逗号分隔 (如 1,4,8,16)")
    parser.add_argument("--num-requests", type=int, default=None, help="每级请求数")
    parser.add_argument("--max-tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
