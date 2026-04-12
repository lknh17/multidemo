"""
p05 强化学习 GRPO - 主训练脚本（trl GRPOTrainer）

使用 trl 库的 GRPOTrainer 进行 GRPO 训练。
核心思路：对每个 prompt 采样一组响应，用组内相对排名作为优势估计。

支持功能:
1. 三种奖励函数切换
2. Group size 消融实验
3. KL 约束控制
4. Gradient Checkpointing
5. 训练监控和日志

使用方式:
    cd p05_rl_grpo
    # 基础训练
    python train.py

    # 指定奖励函数
    python train.py --reward-type correctness

    # 消融实验
    python train.py --group-size 16 --kl-coef 0.1

    # 从 DPO 模型继续
    python train.py --model-path ../p04_dpo/outputs/dpo/final
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRPOConfig, config
from dataset import create_grpo_dataset, extract_model_answer
from reward import correctness_reward, format_reward, composite_reward


# ============================================================
# 1. 奖励函数包装器
# ============================================================
def create_reward_function(reward_type: str, cfg=None):
    """
    创建 GRPOTrainer 所需的奖励函数。
    
    GRPOTrainer 要求奖励函数签名:
        reward_fn(completions, prompts=None, **kwargs) -> list[float]
    
    我们需要把 ground_truth 传进去，这里用闭包实现。
    """
    def reward_fn(completions, ground_truths=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            # 从 completion 提取文本
            if isinstance(completion, dict):
                text = completion.get("content", completion.get("text", str(completion)))
            elif isinstance(completion, list):
                # chat format: [{"role": "assistant", "content": "..."}]
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)
            
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else 0.0
            
            if reward_type == "correctness":
                r = correctness_reward(text, gt)
            elif reward_type == "format":
                r = format_reward(text)
            elif reward_type == "composite":
                result = composite_reward(text, gt)
                r = result["total"]
            else:
                raise ValueError(f"未知奖励类型: {reward_type}")
            
            rewards.append(r)
        
        return rewards
    
    return reward_fn


# ============================================================
# 2. 训练主流程
# ============================================================
def train(cfg: GRPOConfig, args):
    """执行 GRPO 训练"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    
    # ---- 设置随机种子 ----
    from shared.utils import set_seed
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p05 强化学习 GRPO - 训练")
    print("=" * 60)
    
    # ---- 加载 Tokenizer ----
    print("\n[1/5] 加载 Tokenizer...")
    model_path = args.model_path or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 加载模型 ----
    print("\n[2/5] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  模型参数: {num_params:.2f}B")
    print(f"  模型来源: {model_path}")
    
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient Checkpointing 已启用")
    
    # ---- 准备数据 ----
    print("\n[3/5] 准备数据...")
    data_path = args.data_path or "data/gsm8k_train.jsonl"
    
    if not os.path.exists(data_path):
        print(f"  数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    train_dataset = create_grpo_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_samples=args.max_samples or cfg.max_samples,
    )
    
    # ---- 配置奖励函数 ----
    print("\n[4/5] 配置奖励函数...")
    reward_type = args.reward_type or cfg.reward_type
    print(f"  奖励类型: {reward_type}")
    
    # 提取 ground truth 列表
    ground_truths = [item["ground_truth"] for item in train_dataset.data]
    
    # 创建 trl 可用的奖励函数
    reward_fn = create_reward_function(reward_type)
    
    # ---- 配置 GRPO Trainer ----
    print("\n[5/5] 配置训练...")
    
    group_size = args.group_size or cfg.group_size
    kl_coef = args.kl_coef if args.kl_coef is not None else cfg.kl_coef
    
    training_config = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.learning_rate or cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        # GRPO 特有参数
        num_generations=group_size,
        max_completion_length=cfg.max_new_tokens,
        beta=kl_coef,
    )
    
    # ---- 准备 prompt 列表 ----
    prompts = [item["prompt"] for item in train_dataset.data]
    
    # 创建简单的 dataset dict
    from datasets import Dataset
    train_ds = Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
    })
    
    # ---- 创建 GRPOTrainer ----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    
    # 打印训练信息
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    print(f"\n  训练配置:")
    print(f"    数据集大小:     {len(train_dataset)}")
    print(f"    Group Size:     {group_size}")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps} = {effective_batch}")
    print(f"    学习率:         {args.learning_rate or cfg.learning_rate}")
    print(f"    KL 系数:        {kl_coef}")
    print(f"    Clip Ratio:     {cfg.clip_ratio}")
    print(f"    奖励函数:       {reward_type}")
    print(f"    输出目录:       {cfg.output_dir}")
    
    # ---- 训练 ----
    trainer.train()
    
    # ---- 保存最终模型 ----
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # 保存训练配置
    config_path = os.path.join(final_dir, "grpo_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "reward_type": reward_type,
            "group_size": group_size,
            "kl_coef": kl_coef,
            "clip_ratio": cfg.clip_ratio,
            "model_source": model_path,
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  GRPO 训练完成！")
    print(f"  模型已保存到: {final_dir}")
    print("  下一步: python inference.py 对比效果")
    print("=" * 60)


# ============================================================
# 3. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p05 GRPO 训练")
    parser.add_argument("--model-path", type=str, default=None,
                       help="基础模型路径（SFT/DPO 后的模型）")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--reward-type", type=str, default=None,
                       choices=["correctness", "format", "composite"])
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    train(config, args)


if __name__ == "__main__":
    main()
