"""
对比 Base 模型 vs 训练后模型的权重改变量。

输出：
- 全局相对变化（Frobenius norm 百分比）
- 按模块分组的变化（embedding / attention / mlp / norm / lm_head）
- 变化最大的 Top-10 参数
"""

import os
import sys
import argparse
import torch
from collections import defaultdict


def classify(name: str) -> str:
    """把参数名归类到模块大类"""
    n = name.lower()
    if "embed" in n:
        return "embedding"
    if "lm_head" in n:
        return "lm_head"
    if "norm" in n or "layernorm" in n:
        return "norm"
    if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    if any(k in n for k in ["gate_proj", "up_proj", "down_proj", "mlp"]):
        return "mlp"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--trained", type=str, default="outputs/pretrain/final")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"加载 Base   模型: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"加载 Trained 模型: {args.trained}")
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    base_params = dict(base.named_parameters())
    trained_params = dict(trained.named_parameters())

    total_diff_sq = 0.0      # 差值的平方和
    total_base_sq = 0.0      # base 权重的平方和
    total_numel = 0
    max_abs_max = 0.0        # 所有参数中单元素最大改动

    group_diff_sq = defaultdict(float)
    group_base_sq = defaultdict(float)
    group_numel = defaultdict(int)

    per_param = []  # (rel_change, name, shape)

    for name, p_base in base_params.items():
        if name not in trained_params:
            print(f"  跳过 (不在 trained): {name}")
            continue
        p_trained = trained_params[name]
        if p_base.shape != p_trained.shape:
            print(f"  跳过 (shape 不一致): {name}")
            continue

        diff = (p_base.float() - p_trained.float())
        diff_sq = diff.pow(2).sum().item()
        base_sq = p_base.float().pow(2).sum().item()
        numel = p_base.numel()

        total_diff_sq += diff_sq
        total_base_sq += base_sq
        total_numel += numel
        max_abs_max = max(max_abs_max, diff.abs().max().item())

        grp = classify(name)
        group_diff_sq[grp] += diff_sq
        group_base_sq[grp] += base_sq
        group_numel[grp] += numel

        rel = (diff_sq ** 0.5) / (base_sq ** 0.5 + 1e-12) * 100
        per_param.append((rel, name, tuple(p_base.shape), diff_sq ** 0.5, base_sq ** 0.5))

    global_rel = (total_diff_sq ** 0.5) / (total_base_sq ** 0.5 + 1e-12) * 100

    print("\n" + "=" * 70)
    print(f"  全局权重相对变化 (‖ΔW‖ / ‖W‖): {global_rel:.4f}%")
    print(f"  参数总数:                      {total_numel/1e6:.2f} M")
    print(f"  单元素最大改动绝对值:          {max_abs_max:.6f}")
    print("=" * 70)

    # 按模块分组
    print("\n按模块分组的相对变化：")
    print(f"  {'模块':<15}{'参数量(M)':>12}{'‖W‖':>14}{'‖ΔW‖':>14}{'相对变化':>12}")
    print("  " + "-" * 67)
    for grp in sorted(group_numel.keys()):
        w = group_base_sq[grp] ** 0.5
        dw = group_diff_sq[grp] ** 0.5
        rel = dw / (w + 1e-12) * 100
        print(f"  {grp:<15}{group_numel[grp]/1e6:>12.2f}{w:>14.2f}{dw:>14.4f}{rel:>11.4f}%")

    # Top 10 变化最大的参数
    per_param.sort(key=lambda x: x[0], reverse=True)
    print("\n变化最大的 Top-10 参数：")
    print(f"  {'相对变化':>10}  {'‖ΔW‖':>10}  {'‖W‖':>10}  参数名 (shape)")
    print("  " + "-" * 80)
    for rel, name, shape, dw, w in per_param[:10]:
        print(f"  {rel:>9.4f}%  {dw:>10.4f}  {w:>10.2f}  {name} {shape}")

    print("\n变化最小的 Top-5 参数：")
    for rel, name, shape, dw, w in per_param[-5:]:
        print(f"  {rel:>9.4f}%  {dw:>10.4f}  {w:>10.2f}  {name} {shape}")

    # 判定
    print("\n" + "=" * 70)
    if global_rel < 0.5:
        verdict = "几乎没学到东西（学习率太小或数据太少）"
    elif global_rel < 3.0:
        verdict = "✅ 健康区间，学到了但不至于大面积遗忘"
    elif global_rel < 6.0:
        verdict = "⚠️ 较大改动，可能出现指令遵循/对话能力下降"
    else:
        verdict = "❌ 严重改动，基本破坏原模型的通用能力"
    print(f"  结论: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
