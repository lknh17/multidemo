"""
p07 模型量化 - bitsandbytes 量化

使用 bitsandbytes 实现 NF4/FP4 量化：
1. 配置 BitsAndBytesConfig
2. 直接以量化精度加载模型（无需校准数据）
3. 可选：双重量化进一步压缩
4. 保存或直接推理

使用方式:
    python quantize_bnb.py
    python quantize_bnb.py --quant-type nf4 --double-quant
    python quantize_bnb.py --quant-type fp4
    python quantize_bnb.py --8bit
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, bnb_config


def get_bnb_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
):
    """构造 BitsAndBytesConfig"""
    from transformers import BitsAndBytesConfig
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=dtype_map.get(compute_dtype, torch.bfloat16),
        bnb_4bit_use_double_quant=double_quant,
    )


def quantize_bnb(
    model_name: str = None,
    load_in_4bit: bool = None,
    load_in_8bit: bool = None,
    quant_type: str = None,
    compute_dtype: str = None,
    double_quant: bool = None,
    output_dir: str = None,
    test_inference: bool = True,
):
    """执行 bitsandbytes 量化加载"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 使用默认配置或命令行参数
    model_name = model_name or config.model_name
    load_in_8bit = load_in_8bit if load_in_8bit is not None else bnb_config.load_in_8bit
    load_in_4bit = load_in_4bit if load_in_4bit is not None else bnb_config.load_in_4bit
    quant_type = quant_type or bnb_config.bnb_4bit_quant_type
    compute_dtype = compute_dtype or bnb_config.bnb_4bit_compute_dtype
    double_quant = double_quant if double_quant is not None else bnb_config.bnb_4bit_use_double_quant
    output_dir = output_dir or bnb_config.output_dir
    
    if load_in_8bit:
        load_in_4bit = False
        mode_str = "8-bit"
    else:
        mode_str = f"4-bit ({quant_type})"
    
    print("=" * 60)
    print("bitsandbytes 量化")
    print("=" * 60)
    print(f"  模型:       {model_name}")
    print(f"  量化模式:   {mode_str}")
    print(f"  计算精度:   {compute_dtype}")
    print(f"  双重量化:   {double_quant}")
    print(f"  输出目录:   {output_dir}")
    print("=" * 60)
    
    # 1. 构造量化配置
    print("\n[1/3] 构造量化配置...")
    quantization_config = get_bnb_config(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        double_quant=double_quant,
    )
    print(f"  BitsAndBytesConfig: {quantization_config}")
    
    # 2. 以量化精度加载模型
    print("\n[2/3] 以量化精度加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=config.trust_remote_code
    )
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
    )
    load_time = time.time() - start_time
    
    # 统计显存
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"  GPU 显存占用: {vram_used:.2f} GB")
    
    print(f"  加载耗时: {load_time:.1f}s")
    
    # 3. 测试推理
    if test_inference:
        print("\n[3/3] 测试推理...")
        test_prompts = [
            "人工智能的发展前景是",
            "量化技术对大模型的意义在于",
            "深度学习中常见的优化方法包括",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n  Prompt: {prompt}")
            print(f"  Output: {response[:100]}...")
    
    # 保存（bitsandbytes 模型需要特殊处理）
    print(f"\n[保存] bitsandbytes 量化模型...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    model_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / (1024 ** 3)
    
    print(f"\n✅ bitsandbytes 量化完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  模型大小: {model_size:.2f} GB")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="bitsandbytes 量化")
    parser.add_argument("--model", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--quant-type", type=str, default=None, choices=["nf4", "fp4"], help="4-bit 量化类型")
    parser.add_argument("--8bit", action="store_true", dest="use_8bit", help="使用 8-bit 量化")
    parser.add_argument("--double-quant", action="store_true", default=None, help="双重量化")
    parser.add_argument("--no-double-quant", action="store_true", help="禁用双重量化")
    parser.add_argument("--compute-dtype", type=str, default=None, help="计算精度")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no-test", action="store_true", help="跳过推理测试")
    args = parser.parse_args()
    
    double_quant = None
    if args.double_quant:
        double_quant = True
    elif args.no_double_quant:
        double_quant = False
    
    quantize_bnb(
        model_name=args.model,
        load_in_8bit=args.use_8bit,
        quant_type=args.quant_type,
        compute_dtype=args.compute_dtype,
        double_quant=double_quant,
        output_dir=args.output_dir,
        test_inference=not args.no_test,
    )


if __name__ == "__main__":
    main()
