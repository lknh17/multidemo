"""
p02 继续预训练 - 数据下载脚本

下载中文预训练语料：
1. wikimedia/wikipedia — 新版 Wikipedia（长文本、高质量）
2. 回退方案：CLUE 数据集合并

使用方式:
    cd p02_continual_pretrain
    python download_data.py
    python download_data.py --max-samples 10000  # 快速测试
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PretrainConfig, config


# ============================================================
# 1. 数据下载（优选 Wikipedia 长文本）
# ============================================================
def download_wikipedia_zh(max_samples: int = 10000, cache_dir: str = None):
    """
    下载 wikimedia/wikipedia 中文数据集（新版，稳定可用）。
    
    为什么选 wikimedia/wikipedia？
    - 长文本：每篇百科条目平均上千字，适合预训练
    - 稳定可用：wikimedia 官方维护，2024 年新版
    - 高质量：经过社区编辑审核
    - 格式规范：直接包含 "text" 字段
    
    数据集格式：{"id", "url", "title", "text"}
    """
    from datasets import load_dataset
    
    print("下载 wikimedia/wikipedia 中文数据集...")
    print(f"  最大样本数: {max_samples}")
    
    # 使用 HuggingFace mirror
    if os.environ.get("HF_ENDPOINT"):
        print(f"  镜像: {os.environ['HF_ENDPOINT']}")
    
    # 尝试多个可用的日期版本
    possible_configs = [
        "20231101.zh",   # 2023-11 版本
        "20240101.zh",   # 2024-01 版本
        "20240501.zh",   # 2024-05 版本
    ]
    
    dataset = None
    for config_name in possible_configs:
        try:
            print(f"  尝试版本: {config_name}")
            dataset = load_dataset(
                "wikimedia/wikipedia",
                config_name,
                split="train",
                streaming=True,
            )
            print(f"  ✅ 成功加载: {config_name}")
            break
        except Exception as e:
            print(f"  ⚠️ {config_name} 不可用: {str(e)[:100]}")
            continue
    
    if dataset is None:
        raise RuntimeError(
            "所有 wikimedia/wikipedia 版本都不可用。\n"
            "请检查网络连接，或手动指定可用版本。"
        )
    
    # 取子集，过滤短文本
    samples = []
    for i, item in enumerate(dataset):
        if len(samples) >= max_samples:
            break
        text = item.get("text", "")
        # 只保留长度 > 500 字符的文章（Wikipedia 百科条目一般都很长）
        if len(text) > 500:
            samples.append({
                "text": text,
                "title": item.get("title", ""),
            })
        if (i + 1) % 5000 == 0:
            print(f"  已扫描 {i+1} 条，有效 {len(samples)} 条...")
    
    print(f"\n  ✅ 共获取 {len(samples)} 条有效长文本数据")
    return samples


# ============================================================
# 2. 回退方案：CLUE 数据集
# ============================================================
def download_clue_datasets(max_samples: int = 50000, cache_dir: str = None):
    """
    下载 CLUE 数据集中的多个子集，合并为预训练语料。
    （仅作为 Wikipedia 不可用时的备选方案）
    """
    from datasets import load_dataset
    
    print("下载 CLUE 数据集...")
    print(f"  最大样本数: {max_samples}")
    
    if os.environ.get("HF_ENDPOINT"):
        print(f"  镜像: {os.environ['HF_ENDPOINT']}")
    
    samples = []
    max_per_subset = max_samples // 3
    
    # 1. TNews
    print("\n  [1/3] 下载 TNews（新闻）...")
    try:
        dataset_tnews = load_dataset("clue", "tnews", split="train", streaming=True)
        count = 0
        for item in dataset_tnews:
            if count >= max_per_subset:
                break
            text = item.get("sentence", "")
            if len(text) > 50:
                samples.append({"text": text, "title": item.get("label_desc", "新闻")})
                count += 1
        print(f"    ✅ TNews: {count} 条")
    except Exception as e:
        print(f"    ⚠️ TNews 下载失败: {e}")
    
    # 2. iFlytek
    print("\n  [2/3] 下载 iFlytek（应用描述）...")
    try:
        dataset_iflytek = load_dataset("clue", "iflytek", split="train", streaming=True)
        count = 0
        for item in dataset_iflytek:
            if count >= max_per_subset:
                break
            text = item.get("sentence", "")
            if len(text) > 50:
                samples.append({"text": text, "title": item.get("label_desc", "应用描述")})
                count += 1
        print(f"    ✅ iFlytek: {count} 条")
    except Exception as e:
        print(f"    ⚠️ iFlytek 下载失败: {e}")
    
    # 3. CMNLI
    print("\n  [3/3] 下载 CMNLI（自然语言推理）...")
    try:
        dataset_cmnli = load_dataset("clue", "cmnli", split="train", streaming=True)
        count = 0
        for item in dataset_cmnli:
            if count >= max_per_subset:
                break
            text = item.get("sentence1", "") + " " + item.get("sentence2", "")
            if len(text) > 50:
                samples.append({"text": text, "title": "自然语言推理"})
                count += 1
        print(f"    ✅ CMNLI: {count} 条")
    except Exception as e:
        print(f"    ⚠️ CMNLI 下载失败: {e}")
    
    print(f"\n  ✅ 共获取 {len(samples)} 条有效数据")
    return samples


# ============================================================
# 2. 数据探索
# ============================================================
def explore_data(samples: list, num_show: int = 3):
    """展示数据样本和统计信息"""
    
    print(f"\n{'='*60}")
    print(f"  数据统计")
    print(f"{'='*60}")
    
    lengths = [len(s["text"]) for s in samples]
    print(f"  总条数:     {len(samples)}")
    print(f"  平均长度:   {sum(lengths)/len(lengths):.0f} 字符")
    print(f"  最短:       {min(lengths)} 字符")
    print(f"  最长:       {max(lengths)} 字符")
    print(f"  中位数:     {sorted(lengths)[len(lengths)//2]} 字符")
    
    # 长度分布
    ranges = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]
    print(f"\n  长度分布:")
    for lo, hi in ranges:
        count = sum(1 for l in lengths if lo <= l < hi)
        bar = "█" * (count * 40 // len(lengths))
        hi_str = f"{hi}" if hi != float('inf') else "∞"
        print(f"    {lo:>5}-{hi_str:<5}: {count:>6} ({count/len(lengths)*100:5.1f}%) {bar}")
    
    # 样本展示
    print(f"\n{'='*60}")
    print(f"  数据样本（前 {num_show} 条）")
    print(f"{'='*60}")
    
    for i, s in enumerate(samples[:num_show]):
        print(f"\n  [{i+1}] {s.get('title', '无标题')}")
        text = s["text"][:300] + ("..." if len(s["text"]) > 300 else "")
        print(f"      {text}")


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
    print(f"\n  ✅ 已保存到 {save_path} ({size_mb:.1f} MB)")


# ============================================================
# 4. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载预训练数据")
    # 减少样本数：10000 条长文本 >> 50000 条短句，质量优先
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/wiki_zh.jsonl")
    parser.add_argument("--source", type=str, default="wikipedia",
                       choices=["wikipedia", "clue"],
                       help="数据源: wikipedia(推荐) 或 clue(备选)")
    parser.add_argument("--explore-only", action="store_true",
                       help="只展示已下载数据的统计信息")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p02 继续预训练 - 数据下载")
    print("=" * 60)
    
    if args.explore_only and os.path.exists(args.output):
        samples = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        explore_data(samples)
        return
    
    # 下载（优先 Wikipedia 长文本，失败则回退到 CLUE）
    if args.source == "wikipedia":
        try:
            samples = download_wikipedia_zh(max_samples=args.max_samples)
        except Exception as e:
            print(f"\n  ⚠️ Wikipedia 下载失败: {e}")
            print(f"  自动回退到 CLUE 数据集...")
            samples = download_clue_datasets(max_samples=args.max_samples)
    else:
        samples = download_clue_datasets(max_samples=args.max_samples)
    
    # 探索
    explore_data(samples)
    
    # 保存
    save_data(samples, args.output)
    
    print("\n" + "=" * 60)
    print("  ✅ 数据下载完成！")
    print(f"  下一步: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
