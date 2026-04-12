"""
p09 评测体系 - C-Eval 中文知识评测

实现 C-Eval 评测:
- 涵盖中文各学科知识
- 支持 few-shot 评测
- 逐科目统计准确率

使用方式:
    python eval_ceval.py
    python eval_ceval.py --model outputs/p03_sft/final
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. C-Eval 数据加载
# ============================================================
CHOICES = ["A", "B", "C", "D"]

# 科目中文名称
SUBJECT_ZH = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "chinese_language_and_literature": "中国语言文学",
    "modern_chinese_history": "近代史",
    "college_physics": "大学物理",
    "high_school_mathematics": "高中数学",
    "discrete_mathematics": "离散数学",
    "probability_and_statistics": "概率统计",
    "law": "法学",
    "education_science": "教育学",
}


def load_ceval_data(subject: str, split: str = "val", max_samples: int = 100) -> List[Dict]:
    """
    加载 C-Eval 数据集
    
    格式: {'question': str, 'A': str, 'B': str, 'C': str, 'D': str, 'answer': str}
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("ceval/ceval-exam", subject, split=split)
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            samples.append({
                "question": item["question"],
                "choices": [item["A"], item["B"], item["C"], item["D"]],
                "answer": CHOICES.index(item["answer"]),  # 转换为 0-3
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ 加载 {subject} 失败: {e}")
        print(f"  → 使用模拟数据进行演示")
        return _generate_mock_data(subject, max_samples)


def _generate_mock_data(subject: str, n: int = 20) -> List[Dict]:
    """生成模拟 C-Eval 数据"""
    random.seed(42)
    zh_name = SUBJECT_ZH.get(subject, subject)
    samples = []
    for i in range(n):
        samples.append({
            "question": f"关于{zh_name}的第{i+1}个知识点，以下哪个描述是正确的？",
            "choices": [
                f"选项A：关于{zh_name}概念{i+1}的描述A",
                f"选项B：关于{zh_name}概念{i+1}的描述B",
                f"选项C：关于{zh_name}概念{i+1}的描述C",
                f"选项D：关于{zh_name}概念{i+1}的描述D",
            ],
            "answer": random.randint(0, 3),
        })
    return samples


# ============================================================
# 2. Prompt 构建
# ============================================================
def format_ceval_prompt(question: str, choices: List[str],
                        subject_zh: str = "",
                        few_shot_examples: List[Dict] = None) -> str:
    """
    构建 C-Eval 评测 prompt（中文格式）
    
    格式:
        以下是关于{科目}的单项选择题，请直接给出正确答案的选项。
        题目：...
        A. ...
        B. ...
        C. ...
        D. ...
        答案：
    """
    prompt_parts = []
    
    header = f"以下是关于{subject_zh}的单项选择题，请直接给出正确答案的选项。\n"
    prompt_parts.append(header)
    
    # Few-shot 示例
    if few_shot_examples:
        for ex in few_shot_examples:
            ex_text = f"题目：{ex['question']}\n"
            for j, choice in enumerate(ex["choices"]):
                ex_text += f"{CHOICES[j]}. {choice}\n"
            ex_text += f"答案：{CHOICES[ex['answer']]}\n"
            prompt_parts.append(ex_text)
        prompt_parts.append("")
    
    # 当前问题
    q_text = f"题目：{question}\n"
    for j, choice in enumerate(choices):
        q_text += f"{CHOICES[j]}. {choice}\n"
    q_text += "答案："
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
    """预测 C-Eval 答案（比较 A/B/C/D 的 logit）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    choice_logits = {}
    for choice in CHOICES:
        token_id = tokenizer.encode(choice, add_special_tokens=False)
        if token_id:
            choice_logits[choice] = logits[token_id[0]].item()
    
    predicted = max(choice_logits, key=choice_logits.get)
    return predicted


# ============================================================
# 4. 主评测流程
# ============================================================
def evaluate_ceval(model_path: str, cfg=None) -> Dict:
    """
    运行 C-Eval 评测
    
    返回: {
        'subject_results': {科目: {'correct': N, 'total': N, 'accuracy': float}},
        'overall_accuracy': float,
    }
    """
    if cfg is None:
        cfg = config.ceval
    
    model, tokenizer, device = load_model(model_path, config.device)
    
    results = {}
    total_correct = 0
    total_count = 0
    
    for subject in cfg.subjects:
        zh_name = SUBJECT_ZH.get(subject, subject)
        print(f"\n  📝 评测科目: {zh_name} ({subject})")
        
        # 加载数据
        data = load_ceval_data(subject, max_samples=cfg.max_samples_per_subject)
        if not data:
            continue
        
        # 分离 few-shot 和测试数据
        few_shot = data[:cfg.num_few_shot] if cfg.num_few_shot > 0 else []
        test_data = data[cfg.num_few_shot:]
        
        correct = 0
        for item in tqdm(test_data, desc=f"  {zh_name}", leave=False):
            prompt = format_ceval_prompt(
                item["question"], item["choices"], zh_name, few_shot
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
def print_ceval_results(results: Dict, model_name: str):
    """格式化输出 C-Eval 结果"""
    print(f"\n{'='*60}")
    print(f"  C-Eval 评测结果 — {model_name}")
    print(f"{'='*60}")
    
    print(f"\n  {'科目':<18} {'准确率':>8} {'正确/总数':>12}")
    print(f"  {'-'*42}")
    
    for subject, info in results["subject_results"].items():
        zh_name = info.get("zh_name", subject)
        acc = info["accuracy"]
        cnt = f"{info['correct']}/{info['total']}"
        print(f"  {zh_name:<16} {acc:>8.1%} {cnt:>12}")
    
    print(f"  {'-'*42}")
    print(f"  {'总体':<16} {results['overall_accuracy']:>8.1%} "
          f"{results['total_correct']}/{results['total_count']:>8}")
    print()


# ============================================================
# 6. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="C-Eval 中文知识评测")
    parser.add_argument("--model", type=str, default=config.models.base_model,
                        help="模型路径")
    parser.add_argument("--num-few-shot", type=int, default=config.ceval.num_few_shot,
                        help="Few-shot 示例数量")
    parser.add_argument("--max-samples", type=int, default=config.ceval.max_samples_per_subject,
                        help="每科目最大样本数")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p09 评测体系 — C-Eval 中文知识评测")
    print("=" * 60)
    
    cfg = config.ceval
    cfg.num_few_shot = args.num_few_shot
    cfg.max_samples_per_subject = args.max_samples
    
    results = evaluate_ceval(args.model, cfg)
    print_ceval_results(results, args.model)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
