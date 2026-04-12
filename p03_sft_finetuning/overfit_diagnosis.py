"""
p03 SFT 指令微调 - 过拟合诊断

检测 SFT 训练中的过拟合现象：
1. 训练/验证 loss 曲线对比
2. Distinct-N 多样性指标
3. 重复率检测
4. 生成文本质量评估

使用方式:
    cd p03_sft_finetuning
    python overfit_diagnosis.py --model-path outputs/sft_lora/final
"""

import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 1. Loss 曲线分析
# ============================================================
def plot_loss_curves(log_dir: str, save_path: str = None):
    """绘制训练/验证 loss 曲线，检测过拟合"""
    import matplotlib.pyplot as plt
    import glob

    # 从 trainer_state.json 读取 loss
    state_files = glob.glob(os.path.join(log_dir, "**/trainer_state.json"), recursive=True)
    if not state_files:
        print(f"  ⚠️ 未找到 trainer_state.json: {log_dir}")
        return

    with open(state_files[0], "r") as f:
        state = json.load(f)

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in state.get("log_history", []):
        if "loss" in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        print("  ⚠️ 未找到训练 loss 记录")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_steps, train_losses, label="Train Loss", color="blue", alpha=0.8)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color="red", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT 训练/验证 Loss 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 检测过拟合（eval loss 持续上升）
    if len(eval_losses) >= 3:
        recent = eval_losses[-3:]
        if recent[-1] > recent[0] * 1.05:
            ax.annotate("⚠️ 可能过拟合!", xy=(eval_steps[-1], eval_losses[-1]),
                       fontsize=12, color="red", fontweight="bold")

    save_path = save_path or "outputs/loss_curves.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Loss 曲线已保存: {save_path}")


# ============================================================
# 2. Distinct-N 多样性指标
# ============================================================
def compute_distinct_n(texts: list, n: int = 2) -> float:
    """
    计算 Distinct-N 指标（文本多样性）。
    Distinct-N = 不同 n-gram 数 / 总 n-gram 数
    值越高表示生成文本越多样，越低表示越多重复。
    """
    all_ngrams = []
    for text in texts:
        tokens = list(text)  # 字级别 n-gram
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i+n]))

    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ============================================================
# 3. 重复率检测
# ============================================================
def compute_repetition_rate(text: str, window: int = 10) -> float:
    """
    计算文本的重复率。
    在滑动窗口内检测连续重复的子串。
    """
    if len(text) < window * 2:
        return 0.0
    tokens = list(text)
    total_windows = len(tokens) - window + 1
    repeated = 0
    for i in range(total_windows - 1):
        w1 = tokens[i:i+window]
        w2 = tokens[i+1:i+1+window]
        if w1 == w2:
            repeated += 1
    return repeated / max(total_windows - 1, 1)


# ============================================================
# 4. 生成质量诊断
# ============================================================
def diagnose_generation(model_path: str, prompts: list = None):
    """对模型生成文本进行过拟合诊断"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "请解释什么是机器学习",
            "用简单的语言描述量子计算",
            "列举深度学习的三个应用",
            "比较CNN和RNN的区别",
            "什么是注意力机制",
        ]

    print(f"\n[生成诊断] 模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)

    # 计算指标
    d1 = compute_distinct_n(outputs, n=1)
    d2 = compute_distinct_n(outputs, n=2)
    d3 = compute_distinct_n(outputs, n=3)
    rep_rates = [compute_repetition_rate(t) for t in outputs]
    avg_rep = sum(rep_rates) / max(len(rep_rates), 1)
    avg_len = sum(len(t) for t in outputs) / max(len(outputs), 1)

    print(f"\n  📊 生成质量诊断:")
    print(f"    Distinct-1: {d1:.4f} (>0.9 好)")
    print(f"    Distinct-2: {d2:.4f} (>0.7 好)")
    print(f"    Distinct-3: {d3:.4f} (>0.5 好)")
    print(f"    平均重复率: {avg_rep:.4f} (<0.1 好)")
    print(f"    平均输出长度: {avg_len:.0f} 字符")

    # 过拟合判断
    issues = []
    if d2 < 0.5:
        issues.append("Distinct-2 过低，生成多样性不足")
    if avg_rep > 0.2:
        issues.append("重复率过高，模型可能退化")
    if avg_len < 20:
        issues.append("输出过短，模型可能崩溃")

    if issues:
        print(f"\n  ⚠️ 过拟合风险:")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"\n  ✅ 未检测到明显过拟合")

    # 输出样例
    print(f"\n  📝 生成样例:")
    for p, o in zip(prompts[:3], outputs[:3]):
        print(f"    Prompt: {p}")
        print(f"    Output: {o[:100]}...")
        print()

    return {"distinct_1": d1, "distinct_2": d2, "distinct_3": d3,
            "avg_rep": avg_rep, "avg_len": avg_len, "issues": issues}


# ============================================================
# 5. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 过拟合诊断")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  p03 SFT 指令微调 - 过拟合诊断")
    print("=" * 60)

    log_dir = args.log_dir or config.output_dir
    if os.path.exists(log_dir):
        plot_loss_curves(log_dir)

    if not args.plot_only and args.model_path:
        diagnose_generation(args.model_path)

    print("\n" + "=" * 60)
    print("  诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
