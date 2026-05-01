"""
p07 模型量化 - GPTQ 量化

使用 auto-gptq 库实现 GPTQ 量化流程：
1. 加载原始 FP16 模型
2. 准备校准数据
3. GPTQ 量化（逐层 Hessian + 最优权重调整）
4. 保存量化模型

使用方式:
    python quantize_gptq.py
    python quantize_gptq.py --bits 4 --group-size 128
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, gptq_config


def prepare_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    split: str = "train",
    n_samples: int = 128,
    seq_length: int = 512,
    seed: int = 42,
):
    """准备 GPTQ 校准数据集"""
    from datasets import load_dataset
    
    print(f"[校准数据] 加载 {dataset_name} (split={split})...")
    
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_column = "text"
    else:
        dataset = load_dataset(dataset_name, split=split)
        text_column = "text"
    
    # 过滤空文本
    texts = [item[text_column] for item in dataset if len(item[text_column].strip()) > 50]
    
    # 随机采样
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(texts), min(n_samples * 3, len(texts)), replace=False)
    selected_texts = [texts[i] for i in indices]
    
    # Tokenize — auto_gptq 0.7+ 期望 list[dict] 格式
    calibration_data = []
    for text in selected_texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_length)
        if tokens.input_ids.shape[1] >= seq_length // 2:
            calibration_data.append({
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask,
            })
        if len(calibration_data) >= n_samples:
            break
    
    print(f"[校准数据] 准备完成: {len(calibration_data)} 条, 序列长度={seq_length}")
    return calibration_data


def quantize_gptq(
    model_name: str = None,
    bits: int = None,
    group_size: int = None,
    output_dir: str = None,
    desc_act: bool = None,
    sym: bool = None,
    calibration_samples: int = None,
    calibration_seq_length: int = None,
):
    """执行 GPTQ 量化"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer
    
    # 使用默认配置或命令行参数
    model_name = model_name or config.model_name
    bits = bits or gptq_config.bits
    group_size = group_size or gptq_config.group_size
    output_dir = output_dir or gptq_config.output_dir
    desc_act = desc_act if desc_act is not None else gptq_config.desc_act
    sym = sym if sym is not None else gptq_config.sym
    calibration_samples = calibration_samples or config.calibration_samples
    calibration_seq_length = calibration_seq_length or config.calibration_seq_length
    
    print("=" * 60)
    print("GPTQ 量化")
    print("=" * 60)
    print(f"  模型:       {model_name}")
    print(f"  量化位数:   {bits}-bit")
    print(f"  分组大小:   {group_size}")
    print(f"  对称量化:   {sym}")
    print(f"  desc_act:   {desc_act}")
    print(f"  校准样本:   {calibration_samples}")
    print(f"  输出目录:   {output_dir}")
    print("=" * 60)
    
    # 1. 加载 Tokenizer
    print("\n[1/4] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=config.trust_remote_code
    )
    
    # 2. 准备校准数据
    print("\n[2/4] 准备校准数据...")
    calibration_data = prepare_calibration_data(
        tokenizer,
        dataset_name=config.calibration_dataset,
        n_samples=calibration_samples,
        seq_length=calibration_seq_length,
        seed=config.seed,
    )
    
    # 3. 加载模型并量化
    print("\n[3/4] 加载模型并执行 GPTQ 量化...")
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        damp_percent=gptq_config.damp_percent,
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=config.trust_remote_code,
    )
    
    start_time = time.time()
    model.quantize(calibration_data)
    quant_time = time.time() - start_time
    print(f"  量化耗时: {quant_time:.1f}s")
    
    # 4. 保存量化模型
    print("\n[4/4] 保存量化模型...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 统计模型大小
    model_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / (1024 ** 3)
    
    print(f"\n✅ GPTQ 量化完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  模型大小: {model_size:.2f} GB")
    print(f"  量化耗时: {quant_time:.1f}s")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="GPTQ 量化")
    parser.add_argument("--model", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--bits", type=int, default=None, choices=[2, 3, 4, 8], help="量化位数")
    parser.add_argument("--group-size", type=int, default=None, help="分组大小")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--desc-act", action="store_true", default=None, help="按激活值排序")
    parser.add_argument("--no-sym", action="store_true", help="使用非对称量化")
    parser.add_argument("--calibration-samples", type=int, default=None, help="校准样本数")
    args = parser.parse_args()
    
    quantize_gptq(
        model_name=args.model,
        bits=args.bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        desc_act=args.desc_act,
        sym=not args.no_sym if args.no_sym else None,
        calibration_samples=args.calibration_samples,
    )


if __name__ == "__main__":
    main()
