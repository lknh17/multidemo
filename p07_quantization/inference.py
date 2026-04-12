"""
p07 模型量化 - 推理对比

加载原始模型和各量化版本，用相同 prompt 对比输出质量。

使用方式:
    python inference.py
    python inference.py --methods fp16 gptq awq bnb
    python inference.py --prompt "请解释什么是量化"
"""

import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, gptq_config, awq_config, bnb_config


# ============================================================
# 默认测试 Prompts
# ============================================================
COMPARE_PROMPTS = [
    "人工智能的核心技术包括",
    "量化技术可以有效降低大模型的",
    "深度学习与传统机器学习的主要区别在于",
    "Transformer 模型的注意力机制",
    "请简要介绍一下中国的四大发明",
]


def load_fp16_model(model_name: str = None):
    """加载 FP16 原始模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = model_name or config.model_name
    print(f"[FP16] 加载 {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_gptq_model(model_dir: str = None):
    """加载 GPTQ 量化模型"""
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer
    
    model_dir = model_dir or gptq_config.output_dir
    print(f"[GPTQ] 加载 {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )
    return model, tokenizer


def load_awq_model(model_dir: str = None):
    """加载 AWQ 量化模型"""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    
    model_dir = model_dir or awq_config.output_dir
    print(f"[AWQ] 加载 {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_bnb_model(model_name: str = None):
    """加载 bitsandbytes NF4 量化模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = model_name or config.model_name
    print(f"[BnB NF4] 加载 {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    gen_time = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    
    return response, n_tokens, gen_time


def compare_outputs(methods: list = None, prompts: list = None, max_new_tokens: int = 100):
    """对比各方法的推理输出"""
    
    methods = methods or ["fp16", "bnb"]
    prompts = prompts or COMPARE_PROMPTS
    
    print("=" * 70)
    print("量化模型推理对比")
    print("=" * 70)
    
    # 加载模型
    models = {}
    for method in methods:
        try:
            if method == "fp16":
                models[method] = load_fp16_model()
            elif method == "gptq":
                models[method] = load_gptq_model()
            elif method == "awq":
                models[method] = load_awq_model()
            elif method == "bnb":
                models[method] = load_bnb_model()
            else:
                print(f"未知方法: {method}")
        except Exception as e:
            print(f"❌ {method} 加载失败: {e}")
    
    if not models:
        print("没有可用模型，请先执行量化")
        return
    
    # 逐 prompt 对比
    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 70}")
        print(f"Prompt {i+1}/{len(prompts)}: {prompt}")
        print("=" * 70)
        
        for method, (model, tokenizer) in models.items():
            response, n_tokens, gen_time = generate_text(
                model, tokenizer, prompt, max_new_tokens
            )
            speed = n_tokens / gen_time if gen_time > 0 else 0
            
            print(f"\n  [{method.upper()}] ({n_tokens} tokens, {speed:.1f} tok/s)")
            print(f"  {response[:200]}")
    
    # 清理
    for method, (model, tokenizer) in models.items():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="量化推理对比")
    parser.add_argument("--methods", nargs="+", default=None, help="对比方法列表")
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成 token 数")
    args = parser.parse_args()
    
    prompts = [args.prompt] if args.prompt else None
    compare_outputs(methods=args.methods, prompts=prompts, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
