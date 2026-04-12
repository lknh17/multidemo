"""
p02 继续预训练 - 策略对比分析

收集不同训练策略的实验结果并生成对比图表：
1. ZeRO-2 vs ZeRO-3 显存和速度对比
2. cosine vs linear 学习率曲线对比
3. Gradient Checkpointing 开/关对比
4. 全参 vs LoRA 效果对比

使用方式:
    cd p02_continual_pretrain
    python compare_strategies.py --results-dir outputs/pretrain
"""

import os
import sys
import json
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_training_logs(log_dir: str) -> list:
    """加载 HuggingFace Trainer 的训练日志"""
    log_files = glob.glob(os.path.join(log_dir, "*/trainer_state.json"))
    
    results = []
    for f in log_files:
        try:
            with open(f, "r") as fp:
                state = json.load(fp)
            
            exp_name = os.path.basename(os.path.dirname(f))
            log_history = state.get("log_history", [])
            
            losses = [(entry["step"], entry["loss"]) 
                      for entry in log_history if "loss" in entry]
            
            results.append({
                "name": exp_name,
                "losses": losses,
                "total_steps": state.get("global_step", 0),
            })
        except Exception as e:
            print(f"  ⚠️ 无法加载 {f}: {e}")
    
    return results


def plot_comparison(results: list, save_path: str = "logs/comparison.png"):
    """生成对比图表"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    if not results:
        print("  ⚠️ 没有找到实验结果")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('p02 继续预训练 - 策略对比', fontsize=14, fontweight='bold')
    
    colors = ['#6C5CE7', '#0984E3', '#EF4444', '#F59E0B', '#10B981', '#EC4899']
    
    # Loss 曲线
    ax = axes[0]
    for i, r in enumerate(results):
        if r["losses"]:
            steps, losses = zip(*r["losses"])
            ax.plot(steps, losses, label=r["name"], color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 实验汇总表
    ax = axes[1]
    ax.axis('off')
    
    table_data = []
    for r in results:
        final_loss = r["losses"][-1][1] if r["losses"] else "N/A"
        table_data.append([
            r["name"],
            r["total_steps"],
            f"{final_loss:.4f}" if isinstance(final_loss, float) else final_loss,
        ])
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["实验", "总步数", "最终 Loss"],
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 对比图已保存到 {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="策略对比分析")
    parser.add_argument("--results-dir", type=str, default="outputs/pretrain")
    parser.add_argument("--save-path", type=str, default="logs/pretrain_comparison.png")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p02 策略对比分析")
    print("=" * 60)
    
    results = load_training_logs(args.results_dir)
    print(f"  找到 {len(results)} 个实验结果")
    
    for r in results:
        print(f"    - {r['name']}: {r['total_steps']} steps")
    
    plot_comparison(results, args.save_path)


if __name__ == "__main__":
    main()
