"""
p06 MLLM 多模态视觉微调 - 数据下载脚本

下载 LLaVA-Instruct-150K 子集（20K 样本）：
1. 下载指令数据（JSON 格式，包含对话和图像路径）
2. 下载 COCO 图像子集
3. 解析并验证数据完整性

使用方式:
    cd p06_mllm_vision
    python download_data.py
    python download_data.py --max-samples 5000  # 快速测试
    python download_data.py --explore-only       # 只查看数据统计
"""

import os
import sys
import argparse
import json
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MLLMConfig, config


# ============================================================
# 1. 下载 LLaVA-Instruct 数据
# ============================================================
def download_llava_instruct(max_samples: int = 20000, cache_dir: str = None):
    """
    下载 LLaVA-Instruct-150K 数据集子集。
    
    LLaVA-Instruct-150K 特点：
    - 基于 COCO 图像生成的指令数据
    - 包含 conversation（多轮对话）、detail_description、complex_reasoning 三种类型
    - 图像来自 COCO train2017
    - GPT-4 生成的高质量对话标注
    """
    from datasets import load_dataset
    
    print("下载 LLaVA-Instruct-150K 数据集...")
    print(f"  最大样本数: {max_samples}")
    
    if os.environ.get("HF_ENDPOINT"):
        print(f"  镜像: {os.environ['HF_ENDPOINT']}")
    
    # 加载 LLaVA-Instruct-150K（只加载 llava_instruct_150k.json 避免多文件格式冲突）
    try:
        dataset = load_dataset(
            "liuhaotian/LLaVA-Instruct-150K",
            data_files="llava_instruct_150k.json",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ⚠️ load_dataset 失败: {e}")
        print("  尝试使用 huggingface_hub 直接下载 JSON...")
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
        )
        import json as _json
        with open(local_path, "r", encoding="utf-8") as f:
            raw_data = _json.load(f)
        # 转为 datasets 兼容格式
        from datasets import Dataset
        for item in raw_data:
            item["id"] = str(item.get("id", ""))
        dataset = Dataset.from_list(raw_data)
    
    # 随机采样子集
    total = len(dataset)
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:max_samples]
    
    samples = []
    for idx in indices:
        item = dataset[idx]
        sample = {
            "id": item.get("id", str(idx)),
            "image": item.get("image", ""),
            "conversations": item.get("conversations", []),
        }
        
        # 验证对话格式
        if len(sample["conversations"]) >= 2:
            samples.append(sample)
        
        if len(samples) % 5000 == 0 and len(samples) > 0:
            print(f"  已处理 {len(samples)} 条...")
    
    print(f"  ✅ 共获取 {len(samples)} 条有效数据")
    return samples


# ============================================================
# 2. 解析对话格式
# ============================================================
def parse_conversations(sample: dict) -> dict:
    """
    解析 LLaVA 对话格式。
    
    LLaVA 对话格式：
    [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "The image shows a cat sitting on a table..."}
    ]
    
    转换为 Qwen2.5-VL 的 messages 格式：
    [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ]
    """
    conversations = sample.get("conversations", [])
    messages = []
    
    for conv in conversations:
        role = "user" if conv["from"] == "human" else "assistant"
        value = conv["value"]
        
        if role == "user":
            # 检查是否包含 <image> 标签
            content = []
            if "<image>" in value:
                content.append({"type": "image"})
                value = value.replace("<image>", "").strip()
                value = value.lstrip("\n").strip()
            if value:
                content.append({"type": "text", "text": value})
            messages.append({"role": role, "content": content})
        else:
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": value}]
            })
    
    return {
        "id": sample.get("id", ""),
        "image": sample.get("image", ""),
        "messages": messages,
    }


# ============================================================
# 3. 数据探索
# ============================================================
def explore_data(samples: list, num_show: int = 3):
    """展示数据样本和统计信息"""
    
    print(f"\n{'='*60}")
    print(f"  数据统计")
    print(f"{'='*60}")
    
    print(f"  总条数:     {len(samples)}")
    
    # 统计对话轮数
    turn_counts = [len(s.get("conversations", [])) // 2 for s in samples]
    turn_counter = Counter(turn_counts)
    print(f"  平均对话轮数: {sum(turn_counts)/len(turn_counts):.1f}")
    
    # 统计回答长度
    answer_lengths = []
    for s in samples:
        for conv in s.get("conversations", []):
            if conv["from"] == "gpt":
                answer_lengths.append(len(conv["value"]))
    
    if answer_lengths:
        print(f"  平均回答长度: {sum(answer_lengths)/len(answer_lengths):.0f} 字符")
        print(f"  最短回答:     {min(answer_lengths)} 字符")
        print(f"  最长回答:     {max(answer_lengths)} 字符")
    
    # 统计是否包含图像
    has_image = sum(1 for s in samples if s.get("image"))
    print(f"  包含图像:     {has_image} ({has_image/len(samples)*100:.1f}%)")
    
    # 对话轮数分布
    print(f"\n  对话轮数分布:")
    for turns in sorted(turn_counter.keys()):
        count = turn_counter[turns]
        bar = "█" * (count * 40 // len(samples))
        print(f"    {turns}轮: {count:>6} ({count/len(samples)*100:5.1f}%) {bar}")
    
    # 样本展示
    print(f"\n{'='*60}")
    print(f"  数据样本（前 {num_show} 条）")
    print(f"{'='*60}")
    
    for i, s in enumerate(samples[:num_show]):
        print(f"\n  [{i+1}] ID: {s.get('id', '无')}")
        print(f"      图像: {s.get('image', '无')}")
        for conv in s.get("conversations", []):
            role = "👤 Human" if conv["from"] == "human" else "🤖 GPT"
            text = conv["value"][:200] + ("..." if len(conv["value"]) > 200 else "")
            print(f"      {role}: {text}")


# ============================================================
# 4. 保存数据
# ============================================================
def save_data(samples: list, save_path: str):
    """保存为 JSON 格式"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  ✅ 已保存到 {save_path} ({size_mb:.1f} MB)")


# ============================================================
# 4.5 下载 COCO 图片（只下载数据集用到的）
# ============================================================
def download_coco_images(samples: list, image_dir: str = "data/images", max_workers: int = 8):
    """
    根据样本中的 image 字段，从 COCO 官方下载对应图片。
    只下载数据集实际用到的图片，而非整个 COCO train2017。
    
    图片 URL 格式: http://images.cocodataset.org/train2017/{filename}
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    os.makedirs(image_dir, exist_ok=True)
    
    # 收集需要下载的图片
    image_files = set()
    for s in samples:
        img = s.get("image", "")
        if img:
            image_files.add(img)
    
    # 过滤已存在的
    to_download = []
    for img in image_files:
        local_path = os.path.join(image_dir, img)
        if not os.path.exists(local_path):
            to_download.append(img)
    
    if not to_download:
        print(f"\n  ✅ 所有 {len(image_files)} 张图片已存在，跳过下载")
        return image_dir
    
    print(f"\n下载 COCO 图片...")
    print(f"  需要下载: {len(to_download)} 张（已存在: {len(image_files) - len(to_download)} 张）")
    print(f"  保存目录: {image_dir}")
    
    # COCO 图片 URL 基础地址
    base_url = "http://images.cocodataset.org/train2017/"
    
    success = 0
    failed = []
    
    def download_one(filename):
        url = base_url + filename
        local_path = os.path.join(image_dir, filename)
        try:
            urllib.request.urlretrieve(url, local_path)
            return True, filename
        except Exception as e:
            return False, filename
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, img): img for img in to_download}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            ok, filename = future.result()
            if ok:
                success += 1
            else:
                failed.append(filename)
            
            if done_count % 100 == 0 or done_count == len(to_download):
                print(f"  进度: {done_count}/{len(to_download)} (成功: {success}, 失败: {len(failed)})")
    
    print(f"\n  ✅ 图片下载完成: 成功 {success} 张, 失败 {len(failed)} 张")
    if failed:
        print(f"  ⚠️ 失败列表（前10）: {failed[:10]}")
    
    return image_dir


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载 MLLM 微调数据")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--output", type=str, default="data/llava_instruct_20k.json")
    parser.add_argument("--explore-only", action="store_true",
                       help="只展示已下载数据的统计信息")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p06 MLLM 多模态视觉微调 - 数据下载")
    print("=" * 60)
    
    if args.explore_only and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            samples = json.load(f)
        explore_data(samples)
        return
    
    # 下载
    samples = download_llava_instruct(max_samples=args.max_samples)
    
    # 探索
    explore_data(samples)
    
    # 解析为 Qwen2.5-VL 格式并保存
    parsed_samples = [parse_conversations(s) for s in samples]
    
    # 保存原始数据
    save_data(samples, args.output)
    
    # 保存解析后数据
    parsed_path = args.output.replace(".json", "_parsed.json")
    save_data(parsed_samples, parsed_path)
    
    # 下载 COCO 图片
    image_dir = os.path.join(os.path.dirname(args.output), "images")
    download_coco_images(samples, image_dir=image_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ 数据下载完成！")
    print(f"  原始数据: {args.output}")
    print(f"  解析数据: {parsed_path}")
    print(f"  图片目录: {image_dir}")
    print(f"  下一步: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
