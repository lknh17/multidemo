"""
p04 DPO 对齐训练 - 偏好数据下载脚本

下载偏好数据集：
1. UltraFeedback — GPT-4 标注的多维度偏好数据
2. HH-RLHF — Anthropic 的人类偏好对话数据

解析为统一的 prompt / chosen / rejected 三元组格式。

使用方式:
    cd p04_dpo_alignment
    python download_data.py
    python download_data.py --dataset ultrafeedback --max-samples 5000
    python download_data.py --dataset hh_rlhf --max-samples 5000
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DPOConfig, config


# ============================================================
# 1. UltraFeedback 数据下载与解析
# ============================================================
def download_ultrafeedback(max_samples: int = 10000):
    """
    下载 UltraFeedback 偏好数据。
    
    UltraFeedback 特点：
    - 包含 ~64K 条 prompt，每条有 4 个模型回复
    - GPT-4 对回复进行多维度评分（helpfulness, honesty, harmlessness, instruction-following）
    - 根据评分构造 chosen/rejected 偏好对
    - 广泛用于 Zephyr, Mistral 等模型的 DPO 训练
    """
    from datasets import load_dataset
    
    print("下载 UltraFeedback 偏好数据...")
    print(f"  最大样本数: {max_samples}")
    
    # HuggingFace 上有处理好的 binarized 版本
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        streaming=True,
        trust_remote_code=True,
    )
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        # 提取 prompt 和 chosen/rejected 回复
        prompt = item.get("prompt", "")
        chosen_messages = item.get("chosen", [])
        rejected_messages = item.get("rejected", [])
        
        # 提取 assistant 回复文本
        chosen_text = ""
        rejected_text = ""
        
        for msg in chosen_messages:
            if msg.get("role") == "assistant":
                chosen_text = msg.get("content", "")
                break
        
        for msg in rejected_messages:
            if msg.get("role") == "assistant":
                rejected_text = msg.get("content", "")
                break
        
        if prompt and chosen_text and rejected_text and chosen_text != rejected_text:
            samples.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "source": "ultrafeedback",
            })
        
        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1} 条...")
    
    print(f"  ✅ 共获取 {len(samples)} 条偏好数据对")
    return samples


# ============================================================
# 2. HH-RLHF 数据下载与解析
# ============================================================
def download_hh_rlhf(max_samples: int = 10000):
    """
    下载 Anthropic HH-RLHF 数据。
    
    HH-RLHF 特点：
    - Anthropic 收集的人类偏好对话数据
    - 包含 helpful（有帮助）和 harmless（无害）两个子集
    - 每条数据是 chosen/rejected 两个完整对话
    - 是 RLHF 研究的经典 benchmark
    """
    from datasets import load_dataset
    
    print("下载 HH-RLHF 偏好数据...")
    print(f"  最大样本数: {max_samples}")
    
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        chosen_text = item.get("chosen", "")
        rejected_text = item.get("rejected", "")
        
        # 解析对话格式：\n\nHuman: ... \n\nAssistant: ...
        prompt, chosen_response = _parse_hh_conversation(chosen_text)
        _, rejected_response = _parse_hh_conversation(rejected_text)
        
        if prompt and chosen_response and rejected_response:
            samples.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "source": "hh_rlhf",
            })
        
        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1} 条...")
    
    print(f"  ✅ 共获取 {len(samples)} 条偏好数据对")
    return samples


def _parse_hh_conversation(text: str):
    """
    解析 HH-RLHF 的对话格式。
    
    格式示例：
    \n\nHuman: What is DPO?\n\nAssistant: DPO is ...
    """
    # 找到最后一轮 Human-Assistant 对
    human_splits = text.split("\n\nHuman: ")
    if len(human_splits) < 2:
        return "", ""
    
    # 最后一轮对话
    last_turn = human_splits[-1]
    parts = last_turn.split("\n\nAssistant: ", 1)
    
    if len(parts) == 2:
        prompt = parts[0].strip()
        response = parts[1].strip()
        return prompt, response
    
    return "", ""


# ============================================================
# 3. 数据探索
# ============================================================
def explore_data(samples: list, num_show: int = 3):
    """展示偏好数据统计信息"""
    
    print(f"\n{'='*60}")
    print(f"  偏好数据统计")
    print(f"{'='*60}")
    
    prompt_lens = [len(s["prompt"]) for s in samples]
    chosen_lens = [len(s["chosen"]) for s in samples]
    rejected_lens = [len(s["rejected"]) for s in samples]
    
    print(f"  总数据对:      {len(samples)}")
    print(f"  Prompt 平均长度:  {sum(prompt_lens)/len(prompt_lens):.0f} 字符")
    print(f"  Chosen 平均长度:  {sum(chosen_lens)/len(chosen_lens):.0f} 字符")
    print(f"  Rejected 平均长度:{sum(rejected_lens)/len(rejected_lens):.0f} 字符")
    
    # 数据来源分布
    sources = {}
    for s in samples:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\n  来源分布:")
    for src, count in sources.items():
        print(f"    {src}: {count} ({count/len(samples)*100:.1f}%)")
    
    # 样本展示
    print(f"\n{'='*60}")
    print(f"  偏好数据样本（前 {num_show} 条）")
    print(f"{'='*60}")
    
    for i, s in enumerate(samples[:num_show]):
        print(f"\n  [{i+1}] Prompt: {s['prompt'][:120]}...")
        print(f"      ✅ Chosen:   {s['chosen'][:100]}...")
        print(f"      ❌ Rejected: {s['rejected'][:100]}...")


# ============================================================
# 4. 保存数据
# ============================================================
def save_data(samples: list, save_path: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  ✅ 已保存到 {save_path} ({size_mb:.1f} MB)")


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载偏好数据")
    parser.add_argument("--dataset", type=str, default="ultrafeedback",
                       choices=["ultrafeedback", "hh_rlhf", "both"])
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/preference.jsonl")
    parser.add_argument("--explore-only", action="store_true",
                       help="只展示已下载数据的统计信息")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p04 DPO 对齐训练 - 偏好数据下载")
    print("=" * 60)
    
    if args.explore_only and os.path.exists(args.output):
        samples = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        explore_data(samples)
        return
    
    # 下载
    all_samples = []
    
    if args.dataset in ("ultrafeedback", "both"):
        samples = download_ultrafeedback(max_samples=args.max_samples)
        all_samples.extend(samples)
    
    if args.dataset in ("hh_rlhf", "both"):
        samples = download_hh_rlhf(max_samples=args.max_samples)
        all_samples.extend(samples)
    
    # 探索
    explore_data(all_samples)
    
    # 保存
    save_data(all_samples, args.output)
    
    print("\n" + "=" * 60)
    print("  ✅ 偏好数据下载完成！")
    print(f"  下一步: python train.py --algorithm dpo")
    print("=" * 60)


if __name__ == "__main__":
    main()
