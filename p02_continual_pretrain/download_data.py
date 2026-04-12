"""
p02 继续预训练 - 数据下载脚本

下载中文预训练语料：
1. Wikipedia 中文 — 高质量百科知识
2. 可选：SkyPile / WanJuan 子集

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
# 1. 数据下载
# ============================================================
def download_wikipedia_zh(max_samples: int = 50000, cache_dir: str = None):
    """
    下载 Wikipedia 中文数据集。
    
    为什么选 Wikipedia？
    - 质量高：经过社区编辑审核
    - 内容多样：覆盖科学/历史/地理/文化等领域
    - 开源免费：无版权问题
    - 体量适中：中文约 100 万篇，取子集可控制训练时间
    """
    from datasets import load_dataset
    
    print("下载 Wikipedia 中文数据集...")
    print(f"  最大样本数: {max_samples}")
    
    # 使用 HuggingFace mirror
    if os.environ.get("HF_ENDPOINT"):
        print(f"  镜像: {os.environ['HF_ENDPOINT']}")
    
    # 流式加载避免一次性下载全部数据
    dataset = load_dataset(
        "wikipedia", "20220301.zh",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    # 取子集
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        # Wikipedia 数据格式：{"id": "...", "title": "...", "text": "..."}
        if len(item["text"]) > 100:  # 过滤太短的文章
            samples.append({
                "text": item["text"],
                "title": item.get("title", ""),
            })
        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i+1} 条...")
    
    print(f"  ✅ 共获取 {len(samples)} 条有效数据")
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
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/wiki_zh.jsonl")
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
    
    # 下载
    samples = download_wikipedia_zh(max_samples=args.max_samples)
    
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
