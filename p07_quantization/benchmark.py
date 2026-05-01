"""
p07 模型量化 - 量化方法 Benchmark

对比所有量化方法的关键指标：
1. 困惑度 (Perplexity) — 模型质量
2. 推理速度 (tokens/s) — 吞吐量
3. 模型大小 (GB) — 存储开销
4. VRAM 占用 (GB) — 显存开销

使用方式:
    python benchmark.py
    python benchmark.py --methods gptq awq bnb
    python benchmark.py --skip-perplexity       # 跳过困惑度（较慢）
"""

import os
import sys
import time
import json
import argparse

# 阻止 autoawq 的 triton kernel 模块加载（triton 2.1 不兼容 autoawq 0.2.9）
# 只 block triton 相关子模块，允许 awq 其他模块正常导入
class _AwqTritonBlocker:
    """拦截 awq.modules.triton 相关模块的导入"""
    def find_module(self, name, path=None):
        if name.startswith('awq.modules.triton') or name.startswith('awq.nn_modules.triton'):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"{name} blocked (triton version incompatible)")

sys.meta_path.insert(0, _AwqTritonBlocker())

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, gptq_config, awq_config, gguf_config, bnb_config, benchmark_config


@dataclass
class BenchmarkResult:
    """单个方法的评测结果"""
    method: str
    bits: str
    model_size_gb: float = 0.0
    vram_gb: float = 0.0
    perplexity: float = 0.0
    tokens_per_sec: float = 0.0
    load_time_sec: float = 0.0
    quantize_time_sec: float = 0.0
    notes: str = ""


def get_model_size(model_dir: str) -> float:
    """计算模型目录大小 (GB)"""
    path = Path(model_dir)
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def measure_vram() -> float:
    """测量当前 GPU 显存占用 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def clear_gpu():
    """清理 GPU 显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()


def get_model_device(model):
    """获取模型所在设备（兼容不同模型类型）"""
    if hasattr(model, "device"):
        return model.device
    # AWQ / 其他自定义模型：从第一个参数获取 device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_perplexity(model, tokenizer, dataset_name="wikitext", n_samples=256, max_length=512):
    """计算困惑度"""
    from datasets import load_dataset
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][:n_samples]
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = get_model_device(model)
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            seq_len = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def measure_inference_speed(
    model, tokenizer,
    input_length=128, output_length=64,
    warmup_runs=3, benchmark_runs=10,
):
    """测量推理速度 (tokens/s)"""
    
    # 生成固定输入
    prompt = "人工智能技术的快速发展正在改变" * (input_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_length)
    device = get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=output_length, do_sample=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    total_tokens = 0
    start_time = time.time()
    
    for _ in range(benchmark_runs):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=output_length, do_sample=False)
        total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed
    
    return tokens_per_sec


def benchmark_fp16(model_name: str = None) -> BenchmarkResult:
    """评测 FP16 基线"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = model_name or config.model_name
    result = BenchmarkResult(method="FP16 (基线)", bits="16-bit")
    
    clear_gpu()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    result.load_time_sec = time.time() - start
    result.vram_gb = measure_vram()
    
    result.perplexity = compute_perplexity(model, tokenizer, n_samples=benchmark_config.eval_perplexity and 256 or 0)
    result.tokens_per_sec = measure_inference_speed(model, tokenizer)
    
    del model
    clear_gpu()
    
    return result


def benchmark_gptq(model_dir: str = None) -> BenchmarkResult:
    """评测 GPTQ 量化模型"""
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer
    
    model_dir = model_dir or gptq_config.output_dir
    result = BenchmarkResult(method="GPTQ", bits=f"{gptq_config.bits}-bit")
    result.model_size_gb = get_model_size(model_dir)
    
    if not os.path.exists(model_dir):
        result.notes = "模型文件不存在，请先运行 quantize_gptq.py"
        return result
    
    clear_gpu()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    start = time.time()
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )
    result.load_time_sec = time.time() - start
    result.vram_gb = measure_vram()
    
    result.perplexity = compute_perplexity(model, tokenizer)
    result.tokens_per_sec = measure_inference_speed(model, tokenizer)
    
    del model
    clear_gpu()
    
    return result


def benchmark_awq(model_dir: str = None) -> BenchmarkResult:
    """评测 AWQ 量化模型（使用 transformers 原生 AWQ 支持）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_dir = model_dir or awq_config.output_dir
    result = BenchmarkResult(method="AWQ", bits=f"{awq_config.bits}-bit")
    result.model_size_gb = get_model_size(model_dir)
    
    if not os.path.exists(model_dir):
        result.notes = "模型文件不存在，请先运行 quantize_awq.py"
        return result
    
    clear_gpu()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 使用 transformers 原生加载 AWQ 模型（避免 autoawq triton 兼容问题）
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
    )
    result.load_time_sec = time.time() - start
    result.vram_gb = measure_vram()
    
    result.perplexity = compute_perplexity(model, tokenizer)
    result.tokens_per_sec = measure_inference_speed(model, tokenizer)
    
    del model
    clear_gpu()
    
    return result


def benchmark_bnb(model_name: str = None) -> BenchmarkResult:
    """评测 bitsandbytes NF4 量化"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = model_name or config.model_name
    result = BenchmarkResult(method="bitsandbytes (NF4)", bits="4-bit")
    
    clear_gpu()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Monkey-patch: bnb 0.42 已支持 4-bit on GPU，但 transformers 4.51.3
    # 在 dispatch_model 时会对 bnb<0.43.2 的模型 to() 调用抛异常
    import transformers.modeling_utils as _mu
    _orig_to = _mu.PreTrainedModel.to
    def _patched_to(self, *args, **kwargs):
        if getattr(self, "is_quantized", False):
            return self
        return _orig_to(self, *args, **kwargs)
    _mu.PreTrainedModel.to = _patched_to
    
    try:
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
    finally:
        _mu.PreTrainedModel.to = _orig_to
    result.load_time_sec = time.time() - start
    result.vram_gb = measure_vram()
    
    result.perplexity = compute_perplexity(model, tokenizer)
    result.tokens_per_sec = measure_inference_speed(model, tokenizer)
    
    del model
    clear_gpu()
    
    return result


def print_comparison_table(results: List[BenchmarkResult]):
    """打印对比表格"""
    print("\n" + "=" * 90)
    print("量化方法对比 Benchmark 结果")
    print("=" * 90)
    
    header = f"{'方法':<22} {'位数':<8} {'大小(GB)':>9} {'VRAM(GB)':>9} {'PPL':>8} {'速度(tok/s)':>12} {'加载(s)':>8}"
    print(header)
    print("-" * 90)
    
    for r in results:
        if r.notes:
            print(f"{r.method:<22} {r.bits:<8} {'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>12} {'N/A':>8}  ({r.notes})")
        else:
            print(
                f"{r.method:<22} {r.bits:<8} "
                f"{r.model_size_gb:>8.2f}G "
                f"{r.vram_gb:>8.2f}G "
                f"{r.perplexity:>8.2f} "
                f"{r.tokens_per_sec:>11.1f} "
                f"{r.load_time_sec:>7.1f}s"
            )
    
    print("=" * 90)
    
    # 找最佳
    valid = [r for r in results if not r.notes]
    if valid:
        best_ppl = min(valid, key=lambda r: r.perplexity)
        best_speed = max(valid, key=lambda r: r.tokens_per_sec)
        best_size = min(valid, key=lambda r: r.model_size_gb if r.model_size_gb > 0 else float("inf"))
        best_vram = min(valid, key=lambda r: r.vram_gb if r.vram_gb > 0 else float("inf"))
        
        print(f"\n最佳质量 (PPL):    {best_ppl.method} ({best_ppl.perplexity:.2f})")
        print(f"最快推理:          {best_speed.method} ({best_speed.tokens_per_sec:.1f} tok/s)")
        if best_size.model_size_gb > 0:
            print(f"最小模型:          {best_size.method} ({best_size.model_size_gb:.2f} GB)")
        print(f"最省显存:          {best_vram.method} ({best_vram.vram_gb:.2f} GB)")


def run_benchmark(methods: List[str] = None, skip_perplexity: bool = False):
    """运行完整 benchmark"""
    
    all_methods = ["fp16", "gptq", "awq", "bnb"]
    methods = methods or all_methods
    
    print("=" * 60)
    print("量化方法 Benchmark")
    print("=" * 60)
    print(f"  基座模型: {config.model_name}")
    print(f"  评测方法: {', '.join(methods)}")
    print(f"  困惑度:   {'跳过' if skip_perplexity else '开启'}")
    print("=" * 60)
    
    results = []
    
    for method in methods:
        print(f"\n{'=' * 40}")
        print(f"评测: {method.upper()}")
        print(f"{'=' * 40}")
        
        try:
            if method == "fp16":
                result = benchmark_fp16()
            elif method == "gptq":
                result = benchmark_gptq()
            elif method == "awq":
                result = benchmark_awq()
            elif method == "bnb":
                result = benchmark_bnb()
            else:
                print(f"  未知方法: {method}")
                continue
            
            results.append(result)
            if result.notes:
                print(f"  ⚠️ 跳过: {result.notes}")
            else:
                print(f"  PPL={result.perplexity:.2f}, Speed={result.tokens_per_sec:.1f} tok/s, VRAM={result.vram_gb:.2f}G")
            
        except Exception as e:
            print(f"  ❌ 评测失败: {e}")
            results.append(BenchmarkResult(method=method.upper(), bits="N/A", notes=str(e)))
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存结果
    os.makedirs(benchmark_config.report_dir, exist_ok=True)
    report_path = os.path.join(benchmark_config.report_dir, "benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="量化方法 Benchmark")
    parser.add_argument("--methods", nargs="+", default=None, help="评测方法列表")
    parser.add_argument("--skip-perplexity", action="store_true", help="跳过困惑度计算")
    args = parser.parse_args()
    
    run_benchmark(methods=args.methods, skip_perplexity=args.skip_perplexity)


if __name__ == "__main__":
    main()
