"""
p05 强化学习 GRPO - 训练监控

监控训练过程中的关键指标：
1. 奖励曲线
2. KL 散度
3. 奖励欺骗检测
4. 策略分布变化

使用方式:
    cd p05_rl_grpo
    python monitor.py --log-dir logs/grpo
    python monitor.py --check-hacking outputs/grpo
"""

import os
import sys
import argparse
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 读取训练日志
# ============================================================
def load_training_logs(log_dir: str) -> list:
    """从 trainer_state.json 加载训练历史"""
    state_path = os.path.join(log_dir, "trainer_state.json")
    
    if not os.path.exists(state_path):
        # 尝试在 checkpoint 目录中查找
        checkpoints = glob.glob(os.path.join(log_dir, "checkpoint-*", "trainer_state.json"))
        if checkpoints:
            state_path = sorted(checkpoints)[-1]  # 最新的
        else:
            print(f"  未找到训练日志: {state_path}")
            return []
    
    with open(state_path, "r") as f:
        state = json.load(f)
    
    return state.get("log_history", [])


# ============================================================
# 2. 绘制奖励曲线
# ============================================================
def plot_reward_curves(log_dirs: dict, save_path: str = None):
    """
    绘制多组实验的奖励曲线对比。
    
    Args:
        log_dirs: {实验名: 日志目录} 字典
        save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO 训练监控", fontsize=16, fontweight='bold')
    
    colors = ['#6C5CE7', '#0984E3', '#00B894', '#F59E0B', '#EF4444', '#10B981']
    
    for idx, (name, log_dir) in enumerate(log_dirs.items()):
        logs = load_training_logs(log_dir)
        if not logs:
            continue
        
        color = colors[idx % len(colors)]
        
        # 提取各指标
        steps, rewards, kls, losses = [], [], [], []
        for entry in logs:
            step = entry.get("step", 0)
            if "reward" in entry or "mean_reward" in entry:
                steps.append(step)
                rewards.append(entry.get("reward", entry.get("mean_reward", 0)))
            if "kl" in entry or "kl_divergence" in entry:
                kls.append((step, entry.get("kl", entry.get("kl_divergence", 0))))
            if "loss" in entry:
                losses.append((step, entry["loss"]))
        
        # 奖励曲线
        if rewards:
            axes[0, 0].plot(steps[:len(rewards)], rewards, label=name, color=color, linewidth=2)
        
        # KL 散度
        if kls:
            kl_steps, kl_vals = zip(*kls)
            axes[0, 1].plot(kl_steps, kl_vals, label=name, color=color, linewidth=2)
        
        # Loss 曲线
        if losses:
            loss_steps, loss_vals = zip(*losses)
            axes[1, 0].plot(loss_steps, loss_vals, label=name, color=color, linewidth=2)
    
    # 设置标题和标签
    axes[0, 0].set_title("平均奖励", fontweight='bold')
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("KL 散度", fontweight='bold')
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("KL Divergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("训练 Loss", fontweight='bold')
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 空白面板用于文字说明
    axes[1, 1].axis('off')
    summary_text = "GRPO 训练监控要点:\n\n"
    summary_text += "1. 奖励应稳步上升\n"
    summary_text += "2. KL 应保持在合理范围\n"
    summary_text += "3. Loss 应下降后趋于稳定\n"
    summary_text += "4. 奖励上升但准确率不变\n"
    summary_text += "   → 可能发生奖励欺骗\n"
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=13, verticalalignment='center',
                    fontfamily='sans-serif', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存到: {save_path}")
    else:
        plt.show()


# ============================================================
# 3. 奖励欺骗检测
# ============================================================
def detect_reward_hacking(
    log_dir: str,
    eval_results_path: str = None,
    threshold: float = 0.3,
):
    """
    检测奖励欺骗（Reward Hacking）。
    
    奖励欺骗的特征：
    1. 训练奖励持续上升
    2. 但实际准确率停滞或下降
    3. KL 散度持续增大
    4. 生成文本变得不自然
    
    Args:
        log_dir: 训练日志目录
        eval_results_path: 评估结果文件
        threshold: 判断欺骗的阈值
    """
    print("\n" + "=" * 60)
    print("  奖励欺骗检测")
    print("=" * 60)
    
    logs = load_training_logs(log_dir)
    
    if not logs:
        print("  无训练日志，跳过检测")
        return
    
    # 提取奖励和 KL
    rewards = [e.get("reward", e.get("mean_reward")) for e in logs if "reward" in e or "mean_reward" in e]
    kls = [e.get("kl", e.get("kl_divergence")) for e in logs if "kl" in e or "kl_divergence" in e]
    
    warnings = []
    
    # 检查 1: 奖励是否持续上升但后期加速
    if len(rewards) >= 10:
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]
        
        first_slope = (first_half[-1] - first_half[0]) / len(first_half) if len(first_half) > 1 else 0
        second_slope = (second_half[-1] - second_half[0]) / len(second_half) if len(second_half) > 1 else 0
        
        if second_slope > first_slope * 2 and second_slope > threshold:
            warnings.append(
                f"奖励后半段上升过快 (前: {first_slope:.4f}/step, 后: {second_slope:.4f}/step)"
            )
    
    # 检查 2: KL 散度是否过大
    if kls:
        max_kl = max(kls)
        if max_kl > 10.0:
            warnings.append(f"KL 散度过大 (max={max_kl:.2f})，策略偏离参考模型过多")
        
        # KL 是否持续增长
        if len(kls) >= 5:
            recent_kl = kls[-5:]
            if all(recent_kl[i] < recent_kl[i+1] for i in range(len(recent_kl)-1)):
                warnings.append("KL 散度持续增长，策略可能在 exploitation")
    
    # 检查 3: 奖励方差是否在减小（模型找到了 hack 模式）
    if len(rewards) >= 20:
        early_rewards = rewards[:10]
        late_rewards = rewards[-10:]
        
        early_var = sum((r - sum(early_rewards)/len(early_rewards))**2 for r in early_rewards) / len(early_rewards)
        late_var = sum((r - sum(late_rewards)/len(late_rewards))**2 for r in late_rewards) / len(late_rewards)
        
        if late_var < early_var * 0.1 and late_var < 0.01:
            warnings.append(
                f"奖励方差急剧减小 (早期: {early_var:.4f}, 近期: {late_var:.4f})，可能收敛到固定模式"
            )
    
    # 输出结果
    if warnings:
        print("\n  ⚠️ 检测到潜在奖励欺骗信号:")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")
        print("\n  建议:")
        print("    - 增大 KL 系数 (beta)")
        print("    - 检查生成样本的质量")
        print("    - 使用更鲁棒的奖励函数")
        print("    - 添加多样性正则化")
    else:
        print("\n  ✅ 未检测到明显的奖励欺骗信号")
    
    # 打印统计
    if rewards:
        print(f"\n  奖励统计:")
        print(f"    初始: {rewards[0]:.4f}")
        print(f"    最终: {rewards[-1]:.4f}")
        print(f"    最高: {max(rewards):.4f}")
        print(f"    提升: {rewards[-1] - rewards[0]:.4f}")
    
    if kls:
        print(f"\n  KL 统计:")
        print(f"    初始: {kls[0]:.4f}")
        print(f"    最终: {kls[-1]:.4f}")
        print(f"    最大: {max(kls):.4f}")
    
    return warnings


# ============================================================
# 4. 生成质量抽样检查
# ============================================================
def sample_quality_check(
    model_path: str,
    data_path: str = "data/gsm8k_test.jsonl",
    num_samples: int = 10,
):
    """
    抽样检查模型生成质量，辅助判断奖励欺骗。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dataset import load_jsonl, format_chat_messages, extract_answer, extract_model_answer
    
    print("\n" + "=" * 60)
    print("  生成质量抽样检查")
    print("=" * 60)
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    
    # 加载测试数据
    samples = load_jsonl(data_path)[:num_samples]
    
    correct = 0
    for i, s in enumerate(samples):
        gt = extract_answer(s["answer"])
        messages = format_chat_messages(s["question"])
        
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                temperature=0.7, top_p=0.95, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        model_ans = extract_model_answer(response)
        
        is_correct = model_ans is not None and gt is not None and abs(model_ans - gt) < 1e-5
        if is_correct:
            correct += 1
        
        print(f"\n  [{i+1}] {'✅' if is_correct else '❌'}")
        print(f"      问题: {s['question'][:100]}...")
        print(f"      标准答案: {gt}")
        print(f"      模型答案: {model_ans}")
        print(f"      响应预览: {response[:150]}...")
    
    print(f"\n  准确率: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")


# ============================================================
# 5. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p05 GRPO 训练监控")
    parser.add_argument("--log-dir", type=str, default="outputs/grpo",
                       help="训练日志目录")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                       help="多组实验目录进行对比")
    parser.add_argument("--check-hacking", type=str, default=None,
                       help="检测奖励欺骗")
    parser.add_argument("--quality-check", type=str, default=None,
                       help="模型路径，进行生成质量检查")
    parser.add_argument("--save-plot", type=str, default="logs/grpo_monitor.png")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p05 强化学习 GRPO - 训练监控")
    print("=" * 60)
    
    if args.compare:
        # 多组对比
        log_dirs = {}
        for path in args.compare:
            name = os.path.basename(path)
            log_dirs[name] = path
        plot_reward_curves(log_dirs, save_path=args.save_plot)
    elif args.check_hacking:
        detect_reward_hacking(args.check_hacking)
    elif args.quality_check:
        sample_quality_check(args.quality_check)
    else:
        # 默认：绘制单组实验
        plot_reward_curves({"GRPO": args.log_dir}, save_path=args.save_plot)


if __name__ == "__main__":
    main()
