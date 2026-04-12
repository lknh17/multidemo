"""
p09 评测体系 - GSM8K 数学推理评测

实现 GSM8K 评测:
- 生成 Chain-of-Thought 推理过程
- 从推理结果中提取最终答案
- 计算数学准确率

使用方式:
    python eval_gsm8k.py
    python eval_gsm8k.py --model outputs/p05_rl/final
"""

import os
import sys
import re
import json
import argparse
import random
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. GSM8K 数据加载
# ============================================================
def load_gsm8k_data(split: str = "test", max_samples: int = 200) -> List[Dict]:
    """
    加载 GSM8K 数据集
    
    格式: {'question': str, 'answer': str}
    answer 中包含推理过程和最终答案（#### 后面的数字）
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            # 提取标准答案（#### 后面的数字）
            answer_text = item["answer"]
            final_answer = extract_number(answer_text)
            samples.append({
                "question": item["question"],
                "full_answer": answer_text,
                "answer": final_answer,
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ 加载 GSM8K 失败: {e}")
        print(f"  → 使用模拟数据进行演示")
        return _generate_mock_data(max_samples)


def _generate_mock_data(n: int = 20) -> List[Dict]:
    """生成模拟 GSM8K 数据"""
    random.seed(42)
    problems = [
        ("小明有15个苹果，给了小红3个，又买了7个，现在有多少个？",
         "小明原来有15个苹果\n给了小红3个：15-3=12\n又买了7个：12+7=19\n#### 19", 19),
        ("一本书有200页，小明每天读25页，几天能读完？",
         "一本书200页\n每天读25页\n200÷25=8天\n#### 8", 8),
        ("商店有30个橘子，卖出了12个，又进货了20个，现在有多少？",
         "原来30个\n卖出12个：30-12=18\n进货20个：18+20=38\n#### 38", 38),
        ("小红有50元钱，买了一支15元的笔和一本20元的书，还剩多少？",
         "50-15=35\n35-20=15\n#### 15", 15),
        ("一个班有40人，其中男生比女生多6人，男生有多少人？",
         "(40+6)÷2=23\n#### 23", 23),
    ]
    samples = []
    for i in range(n):
        q, a, num = problems[i % len(problems)]
        samples.append({
            "question": q,
            "full_answer": a,
            "answer": num,
        })
    return samples


# ============================================================
# 2. 答案提取
# ============================================================
def extract_number(text: str) -> Optional[float]:
    """
    从文本中提取最终数字答案
    
    策略:
    1. 优先查找 '#### ' 标记后的数字
    2. 否则提取最后出现的数字
    """
    # 策略1: GSM8K 格式的 #### 标记
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # 策略2: 查找 "答案是/等于/=" 后面的数字
    patterns = [
        r"答案[是为]?\s*(-?[\d,]+\.?\d*)",
        r"[=＝]\s*(-?[\d,]+\.?\d*)\s*$",
        r"结果[是为]?\s*(-?[\d,]+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return float(match.group(1).replace(",", ""))
    
    # 策略3: 最后出现的数字
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None


# ============================================================
# 3. Prompt 构建
# ============================================================
def format_gsm8k_prompt(question: str, cot_prompt: str = "") -> str:
    """构建 GSM8K 评测 prompt"""
    prompt = f"问题：{question}\n\n"
    if cot_prompt:
        prompt += cot_prompt
    prompt += "解答："
    return prompt


# ============================================================
# 4. 模型加载与推理
# ============================================================
def load_model(model_path: str, device: str = "auto"):
    """加载模型和 tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    
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


def generate_answer(model, tokenizer, prompt: str,
                    device: str, max_new_tokens: int = 512,
                    temperature: float = 0.0) -> str:
    """生成数学推理回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    # 只取新生成的部分
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return response


# ============================================================
# 5. 主评测流程
# ============================================================
def evaluate_gsm8k(model_path: str, cfg=None) -> Dict:
    """
    运行 GSM8K 评测
    
    返回: {
        'correct': int,
        'total': int,
        'accuracy': float,
        'details': [{'question': str, 'generated': str, 'predicted': num, 'expected': num, 'correct': bool}],
    }
    """
    if cfg is None:
        cfg = config.gsm8k
    
    model, tokenizer, device = load_model(model_path, config.device)
    
    data = load_gsm8k_data(max_samples=cfg.max_samples)
    
    correct = 0
    details = []
    
    print(f"\n  共 {len(data)} 道数学题")
    
    for item in tqdm(data, desc="  GSM8K 评测"):
        prompt = format_gsm8k_prompt(item["question"], cfg.cot_prompt)
        
        generated = generate_answer(
            model, tokenizer, prompt, device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        
        predicted = extract_number(generated)
        expected = item["answer"]
        
        # 判断是否正确（允许浮点误差）
        is_correct = False
        if predicted is not None and expected is not None:
            is_correct = abs(predicted - expected) < 1e-3
        
        if is_correct:
            correct += 1
        
        details.append({
            "question": item["question"],
            "generated": generated[:200],  # 截断保存
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })
    
    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "details": details,
    }


# ============================================================
# 6. 结果输出
# ============================================================
def print_gsm8k_results(results: Dict, model_name: str):
    """格式化输出 GSM8K 结果"""
    print(f"\n{'='*60}")
    print(f"  GSM8K 数学评测结果 — {model_name}")
    print(f"{'='*60}")
    print(f"\n  准确率: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    
    # 展示几个示例
    print(f"\n  --- 部分示例 ---")
    for i, detail in enumerate(results["details"][:5]):
        status = "✅" if detail["correct"] else "❌"
        print(f"\n  [{status}] 题目: {detail['question'][:50]}...")
        print(f"       预测: {detail['predicted']}  |  正确答案: {detail['expected']}")
    print()


# ============================================================
# 7. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GSM8K 数学推理评测")
    parser.add_argument("--model", type=str, default=config.models.base_model,
                        help="模型路径")
    parser.add_argument("--max-samples", type=int, default=config.gsm8k.max_samples,
                        help="最大评测样本数")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p09 评测体系 — GSM8K 数学推理评测")
    print("=" * 60)
    
    cfg = config.gsm8k
    cfg.max_samples = args.max_samples
    
    results = evaluate_gsm8k(args.model, cfg)
    print_gsm8k_results(results, args.model)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
