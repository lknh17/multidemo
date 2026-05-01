"""
p07 模型量化 - AWQ 量化

使用 autoawq 库实现 AWQ (Activation-aware Weight Quantization) 量化：
1. 加载原始 FP16 模型
2. AWQ 搜索最优缩放因子（保护显著权重）
3. 量化为 4-bit
4. 保存量化模型

使用方式:
    python quantize_awq.py
    python quantize_awq.py --bits 4 --group-size 128
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, awq_config


def quantize_awq(
    model_name: str = None,
    bits: int = None,
    group_size: int = None,
    zero_point: bool = None,
    version: str = None,
    output_dir: str = None,
    calibration_samples: int = None,
    calibration_seq_length: int = None,
):
    """执行 AWQ 量化"""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    
    # 使用默认配置或命令行参数
    model_name = model_name or config.model_name
    bits = bits or awq_config.bits
    group_size = group_size or awq_config.group_size
    zero_point = zero_point if zero_point is not None else awq_config.zero_point
    version = version or awq_config.version
    output_dir = output_dir or awq_config.output_dir
    calibration_samples = calibration_samples or config.calibration_samples
    calibration_seq_length = calibration_seq_length or config.calibration_seq_length
    
    print("=" * 60)
    print("AWQ 量化")
    print("=" * 60)
    print(f"  模型:       {model_name}")
    print(f"  量化位数:   {bits}-bit")
    print(f"  分组大小:   {group_size}")
    print(f"  零点偏移:   {zero_point}")
    print(f"  内核版本:   {version}")
    print(f"  校准样本:   {calibration_samples}")
    print(f"  输出目录:   {output_dir}")
    print("=" * 60)
    
    # 1. 加载模型和 Tokenizer
    print("\n[1/3] 加载模型和 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=config.trust_remote_code
    )
    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
        safetensors=True,
    )
    
    # 2. AWQ 量化
    print("\n[2/3] 执行 AWQ 量化（搜索最优缩放因子 + 量化）...")
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }
    
    # 手动加载校准数据（autoawq 内部不支持 wikitext 的 config name）
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]
    calib_texts = calib_texts[:calibration_samples]
    
    start_time = time.time()
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_samples=calibration_samples,
        max_calib_seq_len=calibration_seq_length,
    )
    quant_time = time.time() - start_time
    print(f"  量化耗时: {quant_time:.1f}s")
    
    # 3. 保存量化模型
    print("\n[3/3] 保存量化模型...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 统计模型大小
    model_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / (1024 ** 3)
    
    print(f"\n✅ AWQ 量化完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  模型大小: {model_size:.2f} GB")
    print(f"  量化耗时: {quant_time:.1f}s")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="AWQ 量化")
    parser.add_argument("--model", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--bits", type=int, default=None, choices=[4], help="量化位数")
    parser.add_argument("--group-size", type=int, default=None, help="分组大小")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--version", type=str, default=None, choices=["GEMM", "GEMV"], help="内核版本")
    parser.add_argument("--calibration-samples", type=int, default=None, help="校准样本数")
    args = parser.parse_args()
    
    quantize_awq(
        model_name=args.model,
        bits=args.bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        version=args.version,
        calibration_samples=args.calibration_samples,
    )


if __name__ == "__main__":
    main()
