"""
p02 继续预训练 - 推理对比脚本

加载预训练前后的模型，用同一组 prompt 对比生成效果，
直观展示继续预训练的收益。

使用方式:
    cd p02_continual_pretrain
    python inference.py
    python inference.py --model-path outputs/pretrain/final
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 对比测试 prompt
# ============================================================
COMPARE_PROMPTS = [
    "中国的四大发明是",
    "量子计算的基本原理是",
    "光合作用是植物利用",
    "人工智能的发展历程可以追溯到",
    "深度学习与传统机器学习的主要区别在于",
    "北京是中国的首都，它位于",
]


def load_model(model_path: str):
    """加载模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """生成文本"""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,        # 惩罚重复 token（>1.0 越大越强）
            no_repeat_ngram_size=4,        # 禁止 4-gram 以上的重复
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="预训练前后对比推理")
    parser.add_argument("--model-path", type=str, default=None,
                       help="训练后模型路径（默认: outputs/pretrain/final）")
    parser.add_argument("--base-model", type=str, default=None,
                       help="原始模型名称")
    args = parser.parse_args()
    
    base_model_name = args.base_model or config.model_name
    trained_path = args.model_path or os.path.join(config.output_dir, "final")
    
    print("=" * 60)
    print("  p02 继续预训练 - 效果对比")
    print("=" * 60)
    
    # 加载原始模型
    print("\n加载原始模型...")
    base_model, base_tokenizer = load_model(base_model_name)
    
    # 加载训练后模型
    has_trained = os.path.exists(trained_path)
    if has_trained:
        print("加载训练后模型...")
        trained_model, trained_tokenizer = load_model(trained_path)
    else:
        print(f"⚠️ 未找到训练后模型: {trained_path}")
        print("  将只展示原始模型的输出")
    
    # 对比生成
    for prompt in COMPARE_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"  📝 Prompt: {prompt}")
        print(f"{'─'*60}")
        
        # 原始模型
        base_output = generate(base_model, base_tokenizer, prompt)
        display = base_output[:200] + ("..." if len(base_output) > 200 else "")
        print(f"  🔵 原始模型: {display}")
        
        # 训练后模型
        if has_trained:
            trained_output = generate(trained_model, trained_tokenizer, prompt)
            display = trained_output[:200] + ("..." if len(trained_output) > 200 else "")
            print(f"  🟢 训练后:   {display}")
    
    print(f"\n{'='*60}")
    print("  对比完成！观察训练后模型在中文知识方面的提升。")
    print("=" * 60)


if __name__ == "__main__":
    main()
