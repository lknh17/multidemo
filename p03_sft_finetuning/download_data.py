"""
p03 SFT 指令微调 - 数据下载脚本

下载并预处理 3 个常用中文 SFT 数据集：
1. Alpaca-Chinese — 中文 Alpaca 指令数据（instruction/input/output 格式）
2. BELLE — 百万级中文指令数据子集
3. Firefly — 流萤中文对话数据

数据保存为统一的 JSONL 格式。

使用方式:
    cd p03_sft_finetuning
    # 下载所有数据集
    python download_data.py --max-samples 50000

    # 只下载某个数据集
    python download_data.py --dataset alpaca --max-samples 10000

    # 查看数据统计
    python download_data.py --explore-only
"""

import os
import sys
import json
import argparse
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 数据集下载配置
# ============================================================
DATASET_CONFIGS = {
    "alpaca": {
        "hf_name": "silk-road/alpaca-data-gpt4-chinese",
        "split": "train",
        "format": "alpaca",  # instruction/input/output
        "description": "中文 Alpaca (GPT-4 生成)",
    },
    "belle": {
        "hf_name": "BelleGroup/train_0.5M_CN",
        "split": "train",
        "format": "belle",  # instruction/output
        "description": "BELLE 50万中文指令数据",
    },
    "firefly": {
        "hf_name": "YeungNLP/firefly-train-1.1M",
        "split": "train",
        "format": "firefly",  # kind/input/target
        "description": "Firefly 流萤 110万中文对话数据",
    },
}


# ============================================================
# 2. 格式转换（统一为 Alpaca 格式）
# ============================================================
def convert_to_alpaca_format(sample: dict, source_format: str) -> dict:
    """
    将不同数据集格式统一转换为 Alpaca 格式。
    
    Alpaca 统一格式:
    {
        "instruction": "用户指令",
        "input": "可选的额外输入",
        "output": "期望的模型输出"
    }
    """
    if source_format == "alpaca":
        # 已经是 Alpaca 格式
        return {
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "output": sample.get("output", ""),
        }
    
    elif source_format == "belle":
        # BELLE 格式: instruction + output
        return {
            "instruction": sample.get("instruction", ""),
            "input": "",
            "output": sample.get("output", ""),
        }
    
    elif source_format == "firefly":
        # Firefly 格式: kind + input + target
        return {
            "instruction": sample.get("input", ""),
            "input": "",
            "output": sample.get("target", ""),
        }
    
    else:
        raise ValueError(f"未知数据格式: {source_format}")


# ============================================================
# 3. 下载并保存数据
# ============================================================
def download_dataset(
    dataset_name: str,
    save_dir: str = "data",
    max_samples: Optional[int] = None,
):
    """
    从 HuggingFace 下载数据集并保存为 JSONL 格式。
    
    Args:
        dataset_name: 数据集名称 (alpaca / belle / firefly)
        save_dir: 保存目录
        max_samples: 最大样本数
    """
    from datasets import load_dataset
    
    ds_cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n[下载] {ds_cfg['description']}")
    print(f"  HuggingFace: {ds_cfg['hf_name']}")
    
    # 流式加载数据
    try:
        dataset = load_dataset(
            ds_cfg["hf_name"],
            split=ds_cfg["split"],
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ⚠️ 在线下载失败: {e}")
        print(f"  尝试生成模拟数据...")
        generate_mock_data(dataset_name, save_dir, max_samples or 1000)
        return
    
    # 转换并保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.jsonl")
    
    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            converted = convert_to_alpaca_format(sample, ds_cfg["format"])
            
            # 过滤空数据
            if not converted["instruction"] and not converted["output"]:
                continue
            
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"  已处理 {count} 条...")
            
            if max_samples and count >= max_samples:
                break
    
    print(f"  ✅ 保存完成: {save_path} ({count} 条)")


# ============================================================
# 4. 生成模拟数据（离线场景）
# ============================================================
def generate_mock_data(
    dataset_name: str,
    save_dir: str = "data",
    num_samples: int = 1000,
):
    """当无法在线下载时，生成模拟 SFT 数据用于流程测试"""
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.jsonl")
    
    # 模拟指令模板
    templates = [
        ("请解释什么是{topic}", "关于{topic}的解释如下：{topic}是一个重要的概念..."),
        ("用简单的语言描述{topic}", "{topic}可以这样理解：..."),
        ("请列举{topic}的三个特点", "{topic}有以下特点：1. ... 2. ... 3. ..."),
        ("比较{a}和{b}的区别", "{a}和{b}的主要区别在于：..."),
        ("请给出一个关于{topic}的示例", "以下是{topic}的一个示例：..."),
    ]
    
    topics = [
        "机器学习", "深度学习", "自然语言处理", "计算机视觉",
        "强化学习", "生成对抗网络", "注意力机制", "Transformer",
        "卷积神经网络", "循环神经网络", "预训练模型", "迁移学习",
        "数据增强", "正则化", "优化算法", "损失函数",
    ]
    
    import random
    random.seed(42)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            tpl = random.choice(templates)
            topic = random.choice(topics)
            topic2 = random.choice(topics)
            
            instruction = tpl[0].format(topic=topic, a=topic, b=topic2)
            output = tpl[1].format(topic=topic, a=topic, b=topic2)
            
            sample = {
                "instruction": instruction,
                "input": "",
                "output": output,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"  ✅ 模拟数据已生成: {save_path} ({num_samples} 条)")


# ============================================================
# 5. 数据探索
# ============================================================
def explore_data(data_dir: str = "data"):
    """查看数据统计信息"""
    import glob
    
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not files:
        print(f"  ⚠️ 数据目录为空: {data_dir}")
        print(f"  请先运行: python download_data.py")
        return
    
    for fpath in files:
        name = os.path.basename(fpath).replace(".jsonl", "")
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        # 统计
        inst_lengths = [len(s.get("instruction", "")) for s in samples]
        out_lengths = [len(s.get("output", "")) for s in samples]
        
        print(f"\n  📊 数据集: {name}")
        print(f"     总条数:       {len(samples)}")
        print(f"     指令平均长度: {sum(inst_lengths)/max(len(inst_lengths),1):.0f} 字符")
        print(f"     输出平均长度: {sum(out_lengths)/max(len(out_lengths),1):.0f} 字符")
        print(f"     指令最长:     {max(inst_lengths) if inst_lengths else 0} 字符")
        print(f"     输出最长:     {max(out_lengths) if out_lengths else 0} 字符")
        
        # 样例
        if samples:
            s = samples[0]
            print(f"     样例指令: {s.get('instruction', '')[:80]}...")
            print(f"     样例输出: {s.get('output', '')[:80]}...")


# ============================================================
# 6. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 SFT 数据下载")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "alpaca", "belle", "firefly"],
                       help="要下载的数据集")
    parser.add_argument("--max-samples", type=int, default=50000,
                       help="每个数据集最大样本数")
    parser.add_argument("--save-dir", type=str, default="data",
                       help="数据保存目录")
    parser.add_argument("--explore-only", action="store_true",
                       help="仅查看已有数据统计")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p03 SFT 指令微调 - 数据下载")
    print("=" * 60)
    
    if args.explore_only:
        explore_data(args.save_dir)
        return
    
    if args.dataset == "all":
        for name in DATASET_CONFIGS:
            download_dataset(name, args.save_dir, args.max_samples)
    else:
        download_dataset(args.dataset, args.save_dir, args.max_samples)
    
    print("\n" + "=" * 60)
    explore_data(args.save_dir)
    print("\n" + "=" * 60)
    print("  ✅ 数据下载完成！")
    print("  下一步: python train.py --method lora")
    print("=" * 60)


if __name__ == "__main__":
    main()
