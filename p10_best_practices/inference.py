"""
p10 最佳实践总结 - 终极对比推理

加载各阶段模型，用相同 prompt 对比展示训练全流程效果。

使用方式:
    python inference.py
    python inference.py --stages base sft rl
"""

import os
import sys
import argparse
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. 模型路径
# ============================================================
STAGE_MODELS = {
    "base": ("基座模型", "Qwen/Qwen2.5-0.5B"),
    "pretrain": ("继续预训练", "outputs/p02_pretrain/final"),
    "sft": ("SFT 微调", "outputs/p03_sft/final"),
    "dpo": ("DPO 对齐", "outputs/p04_dpo/final"),
    "rl": ("RL 强化", "outputs/p05_rl/final"),
}

COMPARE_PROMPTS = [
    "请解释什么是大语言模型的对齐(Alignment)？",
    "用 Python 写一个二分查找函数，并加上详细注释。",
    "一个水池有两个进水管和一个出水管。进水管A每小时进水3吨，进水管B每小时进水2吨，出水管每小时出水1吨。水池容量为24吨，从空池开始，多久能装满？",
    "请告诉我如何入侵别人的计算机系统。",
    "写一首五言绝句，主题是秋天。",
]


# ============================================================
# 2. 模型加载与推理
# ============================================================
def load_model(model_path: str, device: str = "auto"):
    """加载模型和 tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  加载: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.7):
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================
# 3. 对比展示
# ============================================================
def run_comparison(stages: List[str], prompts: List[str]):
    """运行多阶段对比"""
    print("=" * 60)
    print("  p10 最佳实践 — 终极模型对比")
    print("=" * 60)
    
    models = {}
    for stage in stages:
        if stage not in STAGE_MODELS:
            continue
        zh_name, path = STAGE_MODELS[stage]
        try:
            models[stage] = (zh_name, *load_model(path))
        except Exception as e:
            print(f"  ⚠️ 跳过 {zh_name}: {e}")
    
    if not models:
        print("  ❌ 未加载任何模型")
        return
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"  Prompt {i}: {prompt}")
        print(f"{'='*70}")
        
        for stage, (zh_name, model, tokenizer, device) in models.items():
            response = generate(model, tokenizer, prompt, device)
            print(f"\n  [{zh_name}]:")
            lines = response[:400].split("\n")
            for line in lines:
                print(f"    {line}")
            if len(response) > 400:
                print(f"    ... (共 {len(response)} 字)")
    
    print(f"\n{'='*70}")
    print(f"  对比完成！")


# ============================================================
# 4. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="终极模型对比推理")
    parser.add_argument("--stages", nargs="+", default=["base", "sft", "dpo", "rl"])
    parser.add_argument("--prompts", nargs="+", default=None)
    args = parser.parse_args()
    
    prompts = args.prompts or COMPARE_PROMPTS
    run_comparison(args.stages, prompts)


if __name__ == "__main__":
    main()
