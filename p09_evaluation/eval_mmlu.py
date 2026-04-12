"""
p09 评测体系 - MMLU 英文知识评测

实现 MMLU (Massive Multitask Language Understanding) 评测:
- 支持 few-shot 评测
- 逐科目统计准确率
- 输出分科目和总体结果

使用方式:
    python eval_mmlu.py
    python eval_mmlu.py --model outputs/p03_sft/final --num-few-shot 5
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. MMLU 数据加载
# ============================================================
CHOICES = ["A", "B", "C", "D"]

# 科目名称中英对照
SUBJECT_ZH = {
    "abstract_algebra": "抽象代数",
    "anatomy": "解剖学",
    "astronomy": "天文学",
    "college_chemistry": "大学化学",
    "college_mathematics": "大学数学",
    "computer_science": "计算机科学",
    "high_school_physics": "高中物理",
    "machine_learning": "机器学习",
    "world_religions": "世界宗教",
}


def load_mmlu_data(subject: str, split: str = "test", max_samples: int = 100) -> List[Dict]:
    """
    加载 MMLU 数据集
    
    格式: {'question': str, 'choices': [A, B, C, D], 'answer': int}
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("cais/mmlu", subject, split=split)
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            samples.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],  # 0-3 对应 A-D
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ 加载 {subject} 失败: {e}")
        print(f"  → 使用模拟数据进行演示")
        return _generate_mock_data(subject, max_samples)


def _generate_mock_data(subject: str, n: int = 20) -> List[Dict]:
    """生成模拟 MMLU 数据（用于无网络环境演示）"""
    random.seed(42)
    samples = []
    templates = [
        ("Which of the following is correct about {topic}?",
         ["Option A description", "Option B description",
          "Option C description", "Option D description"]),
    ]
    for i in range(n):
        q, choices = templates[0]
        samples.append({
            "question": q.format(topic=f"{subject} concept {i+1}"),
            "choices": choices,
            "answer": random.randint(0, 3),
        })
    return samples


# ============================================================
# 2. Prompt 构建
# ============================================================
def format_mmlu_prompt(question: str, choices: List[str],
                       few_shot_examples: List[Dict] = None) -> str:
    """
    构建 MMLU 评测 prompt
    
    格式:
        The following is a multiple choice question about {subject}.
        Question: ...
        A. ...
        B. ...
        C. ...
        D. ...
        Answer:
    """
    prompt_parts = []
    
    # Few-shot 示例
    if few_shot_examples:
        for ex in few_shot_examples:
            ex_text = f"Question: {ex['question']}\n"
            for j, choice in enumerate(ex["choices"]):
                ex_text += f"{CHOICES[j]}. {choice}\n"
            ex_text += f"Answer: {CHOICES[ex['answer']]}\n"
            prompt_parts.append(ex_text)
        prompt_parts.append("")  # 空行分隔
    
    # 当前问题
    q_text = f"Question: {question}\n"
    for j, choice in enumerate(choices):
        q_text += f"{CHOICES[j]}. {choice}\n"
    q_text += "Answer:"
    prompt_parts.append(q_text)
    
    return "\n".join(prompt_parts)


# ============================================================
# 3. 模型加载与推理
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


def predict_answer(model, tokenizer, prompt: str, device: str) -> str:
    """
    预测 MMLU 答案
    
    方法: 比较 A/B/C/D 四个 token 的 logit，选最高的
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
    
    # 获取 A/B/C/D 对应的 token id 和 logit
    choice_logits = {}
    for choice in CHOICES:
        token_id = tokenizer.encode(choice, add_special_tokens=False)
        if token_id:
            choice_logits[choice] = logits[token_id[0]].item()
    
    # 选 logit 最高的
    predicted = max(choice_logits, key=choice_logits.get)
    return predicted


# ============================================================
# 4. 主评测流程
# ============================================================
def evaluate_mmlu(model_path: str, cfg=None) -> Dict:
    """
    运行 MMLU 评测
    
    返回: {
        'subject_results': {科目: {'correct': N, 'total': N, 'accuracy': float}},
        'overall_accuracy': float,
    }
    """
    if cfg is None:
        cfg = config.mmlu
    
    model, tokenizer, device = load_model(model_path, config.device)
    
    results = {}
    total_correct = 0
    total_count = 0
    
    for subject in cfg.subjects:
        zh_name = SUBJECT_ZH.get(subject, subject)
        print(f"\n  📝 评测科目: {zh_name} ({subject})")
        
        # 加载数据
        data = load_mmlu_data(subject, max_samples=cfg.max_samples_per_subject)
        if not data:
            continue
        
        # 分离 few-shot 示例和测试数据
        few_shot = data[:cfg.num_few_shot] if cfg.num_few_shot > 0 else []
        test_data = data[cfg.num_few_shot:]
        
        correct = 0
        for item in tqdm(test_data, desc=f"  {zh_name}", leave=False):
            prompt = format_mmlu_prompt(
                item["question"], item["choices"], few_shot
            )
            predicted = predict_answer(model, tokenizer, prompt, device)
            expected = CHOICES[item["answer"]]
            
            if predicted == expected:
                correct += 1
        
        total = len(test_data)
        accuracy = correct / total if total > 0 else 0.0
        results[subject] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "zh_name": zh_name,
        }
        total_correct += correct
        total_count += total
        
        print(f"    → 准确率: {accuracy:.1%} ({correct}/{total})")
    
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    
    return {
        "subject_results": results,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
    }


# ============================================================
# 5. 结果输出
# ============================================================
def print_mmlu_results(results: Dict, model_name: str):
    """格式化输出 MMLU 结果"""
    print(f"\n{'='*60}")
    print(f"  MMLU 评测结果 — {model_name}")
    print(f"{'='*60}")
    
    print(f"\n  {'科目':<20} {'准确率':>8} {'正确/总数':>12}")
    print(f"  {'-'*44}")
    
    for subject, info in results["subject_results"].items():
        zh_name = info.get("zh_name", subject)
        acc = info["accuracy"]
        cnt = f"{info['correct']}/{info['total']}"
        print(f"  {zh_name:<18} {acc:>8.1%} {cnt:>12}")
    
    print(f"  {'-'*44}")
    print(f"  {'总体':<18} {results['overall_accuracy']:>8.1%} "
          f"{results['total_correct']}/{results['total_count']:>8}")
    print()


# ============================================================
# 6. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MMLU 英文知识评测")
    parser.add_argument("--model", type=str, default=config.models.base_model,
                        help="模型路径")
    parser.add_argument("--num-few-shot", type=int, default=config.mmlu.num_few_shot,
                        help="Few-shot 示例数量")
    parser.add_argument("--max-samples", type=int, default=config.mmlu.max_samples_per_subject,
                        help="每科目最大样本数")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p09 评测体系 — MMLU 英文知识评测")
    print("=" * 60)
    
    # 更新配置
    cfg = config.mmlu
    cfg.num_few_shot = args.num_few_shot
    cfg.max_samples_per_subject = args.max_samples
    
    # 运行评测
    results = evaluate_mmlu(args.model, cfg)
    
    # 打印结果
    print_mmlu_results(results, args.model)
    
    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
