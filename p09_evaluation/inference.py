"""
p09 评测体系 - 五阶段模型对比推理

加载五个阶段的模型，用相同的 prompt 进行 side-by-side 对比。

使用方式:
    python inference.py
    python inference.py --stages base sft dpo
"""

import os
import sys
import argparse
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. 模型加载
# ============================================================
def load_model(model_path: str, device: str = "auto"):
    """加载模型和 tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.7):
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ============================================================
# 2. 对比推理
# ============================================================
def compare_models(stages: Dict[str, str], prompts: List[str],
                   max_new_tokens: int = 256, temperature: float = 0.7):
    """加载多个阶段模型并逐 prompt 对比"""
    models = {}
    for name, path in stages.items():
        try:
            models[name] = load_model(path, config.inference.device)
        except Exception as e:
            print(f"  ⚠️ 跳过 {name}: {e}")

    if not models:
        print("  ❌ 没有成功加载任何模型")
        return

    stage_zh = {"base": "基座", "pretrain": "预训练", "sft": "SFT", "dpo": "DPO", "rl": "RL"}

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"  Prompt {i}: {prompt}")
        print(f"{'='*70}")
        for name, (model, tokenizer, device) in models.items():
            zh = stage_zh.get(name, name)
            response = generate(model, tokenizer, prompt, device, max_new_tokens, temperature)
            print(f"\n  [{zh}]:")
            print(f"  {response[:300]}")
            if len(response) > 300:
                print(f"  ... (共 {len(response)} 字)")
        print()


# ============================================================
# 3. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="五阶段模型对比推理")
    parser.add_argument("--stages", nargs="+", default=["base", "pretrain", "sft", "dpo", "rl"],
                        help="要对比的阶段")
    parser.add_argument("--prompts", nargs="+", default=None, help="自定义 prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 60)
    print("  p09 评测体系 — 五阶段模型对比推理")
    print("=" * 60)

    all_stages = config.models.all_stages()
    stages = {s: all_stages[s] for s in args.stages if s in all_stages}

    prompts = args.prompts or config.inference.compare_prompts
    compare_models(stages, prompts, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
