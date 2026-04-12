"""
p03 SFT 指令微调 - 推理对比脚本

加载并对比 5 种模型变体的输出：
1. base — 原始基座模型
2. lora — LoRA 微调后
3. qlora — QLoRA 微调后
4. dora — DoRA 微调后
5. full — 全参微调后

使用方式:
    cd p03_sft_finetuning
    python inference.py
    python inference.py --method lora --adapter-path outputs/sft_lora/final
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 对比测试 prompt（中文 SFT 场景）
# ============================================================
COMPARE_PROMPTS = [
    "请解释什么是大语言模型？它的核心原理是什么？",
    "用Python写一个快速排序算法",
    "请将以下英文翻译为中文：The future of AI is collaborative.",
    "列举机器学习中常见的过拟合解决方案",
    "请用简单的语言解释量子纠缠",
    "写一首关于春天的五言绝句",
]


# ============================================================
# 1. 模型加载器
# ============================================================
def load_base_model(model_name: str):
    """加载基座模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def load_lora_model(model_name: str, adapter_path: str):
    """加载 LoRA/QLoRA/DoRA 微调模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer


def load_merged_model(merged_path: str):
    """加载合并后的完整模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


# ============================================================
# 2. 生成函数（ChatML 格式）
# ============================================================
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """使用 ChatML 格式生成回复"""
    import torch

    # 构建 ChatML 格式输入
    chat_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 截断到 <|im_end|>
    if "<|im_end|>" in response:
        response = response[:response.index("<|im_end|>")]

    return response


# ============================================================
# 3. 多模型对比
# ============================================================
def compare_models(args):
    """对比多种模型变体的输出"""
    import torch

    print("=" * 60)
    print("  p03 SFT 指令微调 - 模型对比推理")
    print("=" * 60)

    models = {}

    # 基座模型
    base_name = args.base_model or config.model_name
    print(f"\n加载基座模型: {base_name}")
    models["base"] = load_base_model(base_name)

    # 各种微调变体
    method_paths = {
        "lora": "outputs/sft_lora/final",
        "qlora": "outputs/sft_qlora/final",
        "dora": "outputs/sft_dora/final",
        "full": "outputs/sft_full/final",
    }

    if args.adapter_path:
        method = args.method or "lora"
        method_paths = {method: args.adapter_path}

    if args.merged_path:
        method_paths = {"merged": args.merged_path}

    for method, path in method_paths.items():
        if os.path.exists(path):
            print(f"加载 {method} 模型: {path}")
            try:
                if method in ("lora", "qlora", "dora"):
                    models[method] = load_lora_model(base_name, path)
                else:
                    models[method] = load_merged_model(path)
            except Exception as e:
                print(f"  ⚠️ 加载 {method} 失败: {e}")
        else:
            print(f"  跳过 {method}: {path} 不存在")

    # 对比生成
    for prompt in COMPARE_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"  📝 Prompt: {prompt}")
        print(f"{'─'*60}")

        for name, (model, tokenizer) in models.items():
            output = generate_response(model, tokenizer, prompt)
            display = output[:200] + ("..." if len(output) > 200 else "")
            icons = {"base": "🔵", "lora": "🟢", "qlora": "🟡", "dora": "🟣", "full": "🔴", "merged": "⭐"}
            icon = icons.get(name, "⚪")
            print(f"  {icon} {name:>8}: {display}")

    # 清理显存
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  对比完成！观察各微调方法的输出质量差异。")
    print("=" * 60)


# ============================================================
# 4. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 SFT 推理对比")
    parser.add_argument("--method", type=str, default=None,
                       choices=["lora", "qlora", "dora", "full"])
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()

    compare_models(args)


if __name__ == "__main__":
    main()
