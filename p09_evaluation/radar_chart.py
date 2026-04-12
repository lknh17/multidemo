"""
p09 评测体系 - 雷达图生成

生成五阶段模型在各评测维度上的雷达图对比。

使用方式:
    python radar_chart.py
    python radar_chart.py --demo
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. 数据加载
# ============================================================
def load_eval_results(results_dir: str) -> Dict[str, Dict[str, float]]:
    """从评测结果目录加载各阶段的分数"""
    stages = ["base", "pretrain", "sft", "dpo", "rl"]
    all_results = {}
    for stage in stages:
        stage_dir = os.path.join(results_dir, stage)
        scores = {}
        for fname, dim, key in [
            ("mmlu_results.json", "知识问答", "overall_accuracy"),
            ("gsm8k_results.json", "数学推理", "accuracy"),
        ]:
            path = os.path.join(stage_dir, fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    scores[dim] = json.load(f).get(key, 0.0)
        humaneval_path = os.path.join(stage_dir, "humaneval_results.json")
        if os.path.exists(humaneval_path):
            with open(humaneval_path, "r") as f:
                scores["代码生成"] = json.load(f).get("pass_at_k", {}).get("1", 0.0)
        custom_path = os.path.join(stage_dir, "custom_results.json")
        if os.path.exists(custom_path):
            with open(custom_path, "r") as f:
                overall = json.load(f).get("overall_scores", {})
            scores["对话能力"] = overall.get("流畅性", 0.0)
            scores["安全性"] = overall.get("安全性", 0.0)
        all_results[stage] = scores
    return all_results


def generate_demo_data() -> Dict[str, Dict[str, float]]:
    """生成演示数据"""
    return {
        "base":     {"知识问答": 0.35, "对话能力": 0.20, "数学推理": 0.10, "代码生成": 0.08, "安全性": 0.30},
        "pretrain":  {"知识问答": 0.42, "对话能力": 0.22, "数学推理": 0.12, "代码生成": 0.09, "安全性": 0.32},
        "sft":       {"知识问答": 0.45, "对话能力": 0.65, "数学推理": 0.25, "代码生成": 0.18, "安全性": 0.55},
        "dpo":       {"知识问答": 0.46, "对话能力": 0.72, "数学推理": 0.28, "代码生成": 0.20, "安全性": 0.78},
        "rl":        {"知识问答": 0.48, "对话能力": 0.75, "数学推理": 0.35, "代码生成": 0.22, "安全性": 0.82},
    }


# ============================================================
# 2. 雷达图绘制
# ============================================================
def plot_radar_chart(data, dimensions, output_path, figsize=(10, 10), dpi=150):
    """绘制雷达图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    N = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0A0E17")
    ax.set_facecolor("#0A0E17")

    styles = {
        "base":    ("#94A3B8", "基座模型", 1.5),
        "pretrain": ("#F59E0B", "继续预训练", 2.0),
        "sft":     ("#0984E3", "SFT 微调", 2.0),
        "dpo":     ("#6C5CE7", "DPO 对齐", 2.5),
        "rl":      ("#00B894", "RL 强化", 2.5),
    }

    for stage, scores in data.items():
        values = [scores.get(dim, 0.0) for dim in dimensions] + [scores.get(dimensions[0], 0.0)]
        color, label, lw = styles.get(stage, ("#E2E8F0", stage, 1.5))
        ax.plot(angles, values, "o-", color=color, linewidth=lw, label=label, markersize=6)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=14, color="#E2E8F0")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=10, color="#64748B")
    ax.spines["polar"].set_color("#1A1F2E")
    ax.grid(color="#1A1F2E", linewidth=0.8)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=12, framealpha=0.1)
    for text in legend.get_texts():
        text.set_color("#E2E8F0")
    ax.set_title("五阶段模型评测雷达图", fontsize=18, color="#E2E8F0", pad=30, fontweight="bold")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  雷达图已保存到: {output_path}")


# ============================================================
# 3. 分数表格
# ============================================================
def print_score_table(data, dimensions):
    """打印分数对比表格"""
    names = {"base": "基座", "pretrain": "预训练", "sft": "SFT", "dpo": "DPO", "rl": "RL"}
    print(f"\n{'='*70}")
    print(f"  五阶段评测分数对比")
    print(f"{'='*70}")
    header = f"  {'维度':<10}"
    for stage in data:
        header += f" {names.get(stage, stage):>8}"
    print(header)
    print(f"  {'-'*60}")
    for dim in dimensions:
        row = f"  {dim:<8}"
        vals = []
        for stage in data:
            v = data[stage].get(dim, 0.0)
            vals.append(v)
            row += f" {v:>8.1%}"
        print(row)
    print()


# ============================================================
# 4. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="生成评测雷达图")
    parser.add_argument("--results-dir", type=str, default="outputs/eval")
    parser.add_argument("--output", type=str, default=config.radar.output_path)
    parser.add_argument("--demo", action="store_true", help="使用演示数据")
    args = parser.parse_args()

    print("=" * 60)
    print("  p09 评测体系 — 雷达图生成")
    print("=" * 60)

    dimensions = config.radar.dimensions
    if args.demo:
        data = generate_demo_data()
    else:
        data = load_eval_results(args.results_dir)
        if not any(bool(s) for s in data.values()):
            print("\n  未找到评测结果，使用演示数据")
            data = generate_demo_data()

    print_score_table(data, dimensions)
    try:
        plot_radar_chart(data, dimensions, args.output)
    except ImportError:
        print("  ⚠️ 需要安装 matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
