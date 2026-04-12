"""
p05 强化学习 GRPO - 数据下载脚本

下载 GSM8K 数学推理数据集：
1. 训练集 ~7500 条
2. 测试集 ~1300 条

使用方式:
    cd p05_rl_grpo
    python download_data.py
    python download_data.py --max-samples 1000  # 快速测试
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 1. 下载 GSM8K
# ============================================================
def download_gsm8k(max_samples: int = None):
    """
    下载 GSM8K 数学推理数据集。
    
    为什么选 GSM8K？
    - 小学数学应用题，难度适中
    - 答案有标准格式（#### 后跟最终数值）
    - 验证 RL 在数学推理上的提升效果
    - 社区广泛使用的 benchmark
    """
    from datasets import load_dataset
    
    print("下载 GSM8K 数学推理数据集...")
    
    # 使用 HuggingFace mirror
    if os.environ.get("HF_ENDPOINT"):
        print(f"  镜像: {os.environ['HF_ENDPOINT']}")
    
    # 下载训练集
    dataset = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    
    train_data = []
    for item in dataset["train"]:
        train_data.append({
            "question": item["question"],
            "answer": item["answer"],
        })
        if max_samples and len(train_data) >= max_samples:
            break
    
    test_data = []
    for item in dataset["test"]:
        test_data.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    return train_data, test_data


# ============================================================
# 2. 数据探索
# ============================================================
def explore_data(samples: list, split_name: str = "train", num_show: int = 3):
    """展示数据样本和统计信息"""
    
    print(f"\n{'='*60}")
    print(f"  {split_name} 集统计")
    print(f"{'='*60}")
    
    q_lengths = [len(s["question"]) for s in samples]
    a_lengths = [len(s["answer"]) for s in samples]
    
    print(f"  总条数:       {len(samples)}")
    print(f"  问题平均长度: {sum(q_lengths)/len(q_lengths):.0f} 字符")
    print(f"  答案平均长度: {sum(a_lengths)/len(a_lengths):.0f} 字符")
    
    # 答案中步骤数统计
    step_counts = []
    for s in samples:
        steps = [line for line in s["answer"].split("\n") if line.strip() and "####" not in line]
        step_counts.append(len(steps))
    
    print(f"  平均推理步骤: {sum(step_counts)/len(step_counts):.1f} 步")
    print(f"  最少步骤:     {min(step_counts)} 步")
    print(f"  最多步骤:     {max(step_counts)} 步")
    
    # 答案数值分布
    final_answers = []
    for s in samples:
        if "####" in s["answer"]:
            ans_str = s["answer"].split("####")[-1].strip()
            ans_str = ans_str.replace(",", "")
            try:
                final_answers.append(float(ans_str))
            except ValueError:
                pass
    
    if final_answers:
        print(f"\n  最终答案分布:")
        print(f"    最小值: {min(final_answers):.0f}")
        print(f"    最大值: {max(final_answers):.0f}")
        print(f"    中位数: {sorted(final_answers)[len(final_answers)//2]:.0f}")
    
    # 样本展示
    print(f"\n{'='*60}")
    print(f"  数据样本（前 {num_show} 条）")
    print(f"{'='*60}")
    
    for i, s in enumerate(samples[:num_show]):
        print(f"\n  [{i+1}] 问题:")
        print(f"      {s['question'][:200]}{'...' if len(s['question'])>200 else ''}")
        print(f"      答案:")
        answer_lines = s["answer"].split("\n")
        for line in answer_lines[:5]:
            print(f"      {line}")
        if len(answer_lines) > 5:
            print(f"      ... (共 {len(answer_lines)} 行)")


# ============================================================
# 3. 保存数据
# ============================================================
def save_data(samples: list, save_path: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  已保存到 {save_path} ({size_mb:.1f} MB)")


# ============================================================
# 4. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载 GSM8K 数据")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--explore-only", action="store_true",
                       help="只展示已下载数据的统计信息")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p05 强化学习 GRPO - 数据下载")
    print("=" * 60)
    
    train_path = os.path.join(args.output_dir, "gsm8k_train.jsonl")
    test_path = os.path.join(args.output_dir, "gsm8k_test.jsonl")
    
    if args.explore_only:
        for path, name in [(train_path, "train"), (test_path, "test")]:
            if os.path.exists(path):
                samples = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        samples.append(json.loads(line))
                explore_data(samples, name)
        return
    
    # 下载
    train_data, test_data = download_gsm8k(max_samples=args.max_samples)
    
    # 探索
    explore_data(train_data, "train")
    explore_data(test_data, "test", num_show=2)
    
    # 保存
    print(f"\n保存数据...")
    save_data(train_data, train_path)
    save_data(test_data, test_path)
    
    print("\n" + "=" * 60)
    print("  数据下载完成！")
    print(f"  下一步: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
