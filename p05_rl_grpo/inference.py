"""
p05 强化学习 GRPO - 推理对比脚本

对比 SFT / DPO / RL (GRPO) 三种模型在 GSM8K 上的表现。
测试数学推理准确率和生成质量。

使用方式:
    cd p05_rl_grpo
    python inference.py
    python inference.py --model-path outputs/grpo/final
    python inference.py --compare-all
"""

import os
import sys
import argparse
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from dataset import load_jsonl, format_chat_messages, extract_answer, extract_model_answer
from reward import composite_reward


# ============================================================
# 对比测试 prompt
# ============================================================
MATH_PROMPTS = [
    "小明有 15 个苹果，给了小红 3 个，又买了 7 个，请问小明现在有多少个苹果？",
    "一个教室有 4 排座位，每排 8 个座位。如果有 25 个学生，还剩多少个空座位？",
    "妈妈买了 3 斤苹果，每斤 5.5 元。又买了 2 斤香蕉，每斤 3 元。一共花了多少钱？",
    "火车从 A 站到 B 站需要 3 小时，速度是每小时 120 公里。A 站到 B 站有多远？",
    "小华有 100 元钱，买了 2 本书每本 15 元，又买了 1 支笔 8 元。还剩多少钱？",
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


def generate(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """生成推理回答"""
    import torch
    
    messages = format_chat_messages(question)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 1. GSM8K 准确率测试
# ============================================================
def eval_gsm8k(model, tokenizer, data_path: str, max_samples: int = 100):
    """
    在 GSM8K 测试集上评估准确率。
    """
    print(f"\n[GSM8K 评估] 数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"  测试数据不存在，跳过评估")
        return None
    
    samples = load_jsonl(data_path)[:max_samples]
    
    correct = 0
    total = 0
    reward_sum = 0.0
    results = []
    
    for i, s in enumerate(samples):
        gt = extract_answer(s["answer"])
        if gt is None:
            continue
        
        response = generate(model, tokenizer, s["question"])
        model_ans = extract_model_answer(response)
        
        is_correct = model_ans is not None and abs(model_ans - gt) < 1e-5
        if is_correct:
            correct += 1
        total += 1
        
        # 计算奖励
        r = composite_reward(response, gt)
        reward_sum += r["total"]
        
        results.append({
            "question": s["question"],
            "ground_truth": gt,
            "model_answer": model_ans,
            "correct": is_correct,
            "reward": r["total"],
        })
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(samples)}, 当前准确率: {correct/total*100:.1f}%")
    
    accuracy = correct / total if total > 0 else 0
    avg_reward = reward_sum / total if total > 0 else 0
    
    print(f"\n  评估结果:")
    print(f"    准确率: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"    平均奖励: {avg_reward:.4f}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_reward": avg_reward,
        "results": results,
    }


# ============================================================
# 2. 多模型对比
# ============================================================
def compare_models(model_paths: dict, data_path: str, max_samples: int = 100):
    """对比多个模型"""
    
    print("\n" + "=" * 60)
    print("  模型对比评估")
    print("=" * 60)
    
    results = {}
    
    for name, path in model_paths.items():
        if not os.path.exists(path) and "/" not in path:
            print(f"\n  跳过 {name}: 路径不存在 ({path})")
            continue
        
        print(f"\n{'─'*60}")
        print(f"  加载模型: {name} ({path})")
        print(f"{'─'*60}")
        
        try:
            model, tokenizer = load_model(path)
            result = eval_gsm8k(model, tokenizer, data_path, max_samples)
            if result:
                results[name] = result
            
            # 释放显存
            del model
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  加载失败: {e}")
    
    # 汇总对比
    if results:
        print("\n" + "=" * 60)
        print("  对比汇总")
        print("=" * 60)
        print(f"\n  {'模型':<20} {'准确率':>10} {'平均奖励':>10}")
        print(f"  {'─'*40}")
        for name, r in results.items():
            print(f"  {name:<20} {r['accuracy']*100:>9.1f}% {r['avg_reward']:>10.4f}")
    
    return results


# ============================================================
# 3. 定性展示
# ============================================================
def qualitative_demo(model, tokenizer, model_name: str = "GRPO"):
    """展示模型在数学题上的推理过程"""
    
    print(f"\n{'='*60}")
    print(f"  {model_name} 推理展示")
    print(f"{'='*60}")
    
    for question in MATH_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"  问题: {question}")
        print(f"{'─'*60}")
        
        response = generate(model, tokenizer, question)
        print(f"  回答:")
        for line in response.split("\n"):
            print(f"    {line}")
        
        model_ans = extract_model_answer(response)
        print(f"\n  提取答案: {model_ans}")


# ============================================================
# 4. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p05 GRPO 推理对比")
    parser.add_argument("--model-path", type=str, default=None,
                       help="GRPO 模型路径")
    parser.add_argument("--base-model", type=str, default=None,
                       help="基础模型名称")
    parser.add_argument("--sft-model", type=str, default=None,
                       help="SFT 模型路径")
    parser.add_argument("--dpo-model", type=str, default=None,
                       help="DPO 模型路径")
    parser.add_argument("--test-data", type=str, default="data/gsm8k_test.jsonl")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--compare-all", action="store_true",
                       help="对比所有可用模型")
    parser.add_argument("--demo-only", action="store_true",
                       help="只做定性展示")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p05 强化学习 GRPO - 推理对比")
    print("=" * 60)
    
    if args.compare_all or (args.sft_model or args.dpo_model):
        # 多模型对比
        model_paths = {}
        model_paths["Base"] = args.base_model or config.model_name
        if args.sft_model:
            model_paths["SFT"] = args.sft_model
        if args.dpo_model:
            model_paths["DPO"] = args.dpo_model
        
        grpo_path = args.model_path or os.path.join(config.output_dir, "final")
        if os.path.exists(grpo_path):
            model_paths["GRPO"] = grpo_path
        
        compare_models(model_paths, args.test_data, args.max_samples)
    else:
        # 单模型评估
        model_path = args.model_path or config.model_name
        grpo_path = os.path.join(config.output_dir, "final")
        
        if os.path.exists(grpo_path) and args.model_path is None:
            model_path = grpo_path
        
        print(f"\n  加载模型: {model_path}")
        model, tokenizer = load_model(model_path)
        
        if args.demo_only:
            qualitative_demo(model, tokenizer)
        else:
            # 定量评估
            eval_gsm8k(model, tokenizer, args.test_data, args.max_samples)
            # 定性展示
            qualitative_demo(model, tokenizer)
    
    print(f"\n{'='*60}")
    print("  评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
