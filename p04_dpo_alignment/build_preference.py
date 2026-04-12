"""
p04 DPO 对齐训练 - 偏好数据构造

从 SFT 模型通过 Reject Sampling 构建偏好数据：
1. 对每个 prompt 生成多个回复（采样）
2. 使用评分模型或规则对回复评分/排名
3. 选取最好和最差的回复构成 chosen/rejected 偏好对

这是 RLHF/DPO 实践中的关键步骤——数据质量直接决定对齐效果。

使用方式:
    cd p04_dpo_alignment
    python build_preference.py --model-path ../p03_sft_finetuning/outputs/sft/final
    python build_preference.py --prompts-file data/prompts.jsonl --num-responses 4
"""

import os
import sys
import argparse
import json
import random
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 1. 加载生成模型
# ============================================================
def load_generation_model(model_path: str):
    """加载用于生成回复的 SFT 模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================
# 2. 多次采样生成
# ============================================================
def generate_multiple_responses(
    model, tokenizer, prompt: str,
    num_responses: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> List[str]:
    """
    对同一个 prompt 生成多个不同的回复。
    
    为什么用高 temperature 采样？
    - 低 temperature（接近 0）→ 生成近乎相同的回复
    - 高 temperature（0.7-1.0）→ 生成多样化回复，才能产生质量差异
    - 多样性是构造偏好对的前提
    """
    import torch
    
    # 构造 chat 格式
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    responses = []
    for _ in range(num_responses):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    
    return responses


# ============================================================
# 3. 回复评分（多种策略）
# ============================================================
def score_responses_by_length(responses: List[str]) -> List[float]:
    """
    策略1：基于长度评分（简单基线）。
    
    假设：更详细的回答通常更好（有限度内）。
    这只是一个简单的 baseline，实际应使用 reward model。
    """
    scores = []
    for r in responses:
        length = len(r)
        # 理想长度在 200-800 字符之间，太短不够详细，太长可能啰嗦
        if length < 50:
            score = 0.2
        elif length < 200:
            score = 0.5 + length / 400
        elif length < 800:
            score = 1.0
        else:
            score = max(0.3, 1.0 - (length - 800) / 2000)
        scores.append(score)
    return scores


def score_responses_by_reward_model(
    responses: List[str], prompt: str, reward_model, reward_tokenizer
) -> List[float]:
    """
    策略2：使用 Reward Model 评分。
    
    这是标准做法：训练一个 reward model，对每个回复打分。
    Reward model 通常是在人类标注的偏好数据上训练的。
    """
    import torch
    
    scores = []
    for response in responses:
        text = f"User: {prompt}\nAssistant: {response}"
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(reward_model.device)
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            # 通常 reward model 输出一个标量分数
            if hasattr(outputs, "logits"):
                score = outputs.logits.squeeze().item()
            else:
                score = outputs[0].squeeze().item()
        scores.append(score)
    
    return scores


def score_responses_by_rules(responses: List[str], prompt: str) -> List[float]:
    """
    策略3：基于规则的综合评分。
    
    结合多个维度：
    - 长度适中性
    - 是否包含关键信息
    - 格式规范性（不重复、不截断）
    - 与 prompt 的相关性
    """
    scores = []
    for r in responses:
        score = 0.0
        
        # (1) 长度分数 (0-0.3)
        length = len(r)
        if 100 <= length <= 800:
            score += 0.3
        elif 50 <= length <= 1200:
            score += 0.15
        else:
            score += 0.05
        
        # (2) 不重复分数 (0-0.3)
        sentences = r.split("。")
        unique_sentences = set(s.strip() for s in sentences if len(s.strip()) > 5)
        if len(sentences) > 0:
            repetition_ratio = len(unique_sentences) / max(1, len(sentences))
            score += 0.3 * repetition_ratio
        
        # (3) 完整性分数 (0-0.2)
        if r.endswith(("。", "！", "？", ".", "!", "?")):
            score += 0.2
        elif len(r) > 50:
            score += 0.1
        
        # (4) 相关性分数 (0-0.2)
        prompt_words = set(prompt)
        response_words = set(r[:200])
        overlap = len(prompt_words & response_words) / max(1, len(prompt_words))
        score += 0.2 * min(1.0, overlap)
        
        scores.append(score)
    
    return scores


# ============================================================
# 4. 构造偏好对
# ============================================================
def build_preference_pairs(
    responses: List[str],
    scores: List[float],
    prompt: str,
    strategy: str = "best_worst",
) -> List[Dict]:
    """
    从评分后的回复中构造偏好对。
    
    策略:
    - best_worst: 只取最好和最差的一对
    - all_pairs: 生成所有两两配对（数据量更大）
    - top_bottom_k: 取 top-k 和 bottom-k 的配对
    """
    indexed = list(zip(range(len(responses)), scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    
    pairs = []
    
    if strategy == "best_worst":
        # 只取最好和最差
        best_idx = indexed[0][0]
        worst_idx = indexed[-1][0]
        if indexed[0][1] > indexed[-1][1]:
            pairs.append({
                "prompt": prompt,
                "chosen": responses[best_idx],
                "rejected": responses[worst_idx],
                "chosen_score": indexed[0][1],
                "rejected_score": indexed[-1][1],
            })
    
    elif strategy == "all_pairs":
        # 所有两两配对（分数高的作为 chosen）
        for i in range(len(indexed)):
            for j in range(i + 1, len(indexed)):
                if indexed[i][1] > indexed[j][1] + 0.05:  # 分数差 > 阈值
                    pairs.append({
                        "prompt": prompt,
                        "chosen": responses[indexed[i][0]],
                        "rejected": responses[indexed[j][0]],
                        "chosen_score": indexed[i][1],
                        "rejected_score": indexed[j][1],
                    })
    
    elif strategy == "top_bottom_k":
        # top-2 vs bottom-2
        k = min(2, len(indexed) // 2)
        for i in range(k):
            for j in range(k):
                top_idx = indexed[i][0]
                bot_idx = indexed[-(j+1)][0]
                if indexed[i][1] > indexed[-(j+1)][1]:
                    pairs.append({
                        "prompt": prompt,
                        "chosen": responses[top_idx],
                        "rejected": responses[bot_idx],
                        "chosen_score": indexed[i][1],
                        "rejected_score": indexed[-(j+1)][1],
                    })
    
    return pairs


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="构造偏好数据")
    parser.add_argument("--model-path", type=str, default=None,
                       help="SFT 模型路径")
    parser.add_argument("--prompts-file", type=str, default=None,
                       help="prompt 文件路径（JSONL，每行 {\"prompt\": \"...\"}）")
    parser.add_argument("--num-responses", type=int, default=4,
                       help="每个 prompt 生成的回复数")
    parser.add_argument("--scoring", type=str, default="rules",
                       choices=["length", "rules", "reward_model"],
                       help="评分策略")
    parser.add_argument("--pair-strategy", type=str, default="best_worst",
                       choices=["best_worst", "all_pairs", "top_bottom_k"])
    parser.add_argument("--output", type=str, default="data/preference_built.jsonl")
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p04 偏好数据构造 - Reject Sampling")
    print("=" * 60)
    
    # 默认 prompts
    default_prompts = [
        "什么是机器学习？请简要介绍。",
        "请解释什么是深度学习中的注意力机制。",
        "如何用 Python 实现一个简单的排序算法？",
        "请介绍 Transformer 架构的核心组件。",
        "什么是迁移学习？有哪些常见的方法？",
        "请解释梯度下降算法的基本原理。",
        "什么是正则化？为什么需要正则化？",
        "请介绍自然语言处理的主要任务和应用。",
        "什么是卷积神经网络？它有什么优势？",
        "请解释 RLHF 的基本流程。",
    ]
    
    # 加载 prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        print(f"\n加载 prompts: {args.prompts_file}")
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                prompts.append(item.get("prompt", item.get("text", "")))
        prompts = prompts[:args.max_prompts]
    else:
        print("\n使用默认 prompts（共 10 条示例）")
        prompts = default_prompts
    
    print(f"  Prompt 数量: {len(prompts)}")
    print(f"  每条生成:    {args.num_responses} 个回复")
    print(f"  评分策略:    {args.scoring}")
    print(f"  配对策略:    {args.pair_strategy}")
    
    # 加载模型
    model_path = args.model_path or config.model_name
    print(f"\n加载生成模型...")
    model, tokenizer = load_generation_model(model_path)
    
    # 逐条生成和评分
    all_pairs = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:50]}...")
        
        # 生成多个回复
        responses = generate_multiple_responses(
            model, tokenizer, prompt,
            num_responses=args.num_responses,
            temperature=args.temperature,
        )
        
        # 评分
        if args.scoring == "length":
            scores = score_responses_by_length(responses)
        elif args.scoring == "rules":
            scores = score_responses_by_rules(responses, prompt)
        else:
            print("  ⚠️ Reward model 评分需要额外配置，使用规则评分代替")
            scores = score_responses_by_rules(responses, prompt)
        
        # 展示评分
        for j, (r, s) in enumerate(zip(responses, scores)):
            print(f"  回复{j+1} (score={s:.3f}): {r[:60]}...")
        
        # 构造偏好对
        pairs = build_preference_pairs(responses, scores, prompt, args.pair_strategy)
        all_pairs.extend(pairs)
        print(f"  → 生成 {len(pairs)} 个偏好对")
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"  ✅ 偏好数据构造完成！")
    print(f"  总偏好对: {len(all_pairs)}")
    print(f"  保存路径: {args.output}")
    print(f"  下一步: python train.py --algorithm dpo --data-path {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
