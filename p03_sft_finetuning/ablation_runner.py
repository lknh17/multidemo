"""
p03 SFT 指令微调 - 消融实验运行器

批量运行 LoRA 消融实验，比较不同超参数的效果：
1. LoRA rank 消融: r = 8, 16, 32, 64, 128
2. LoRA alpha 消融: alpha/r = 1, 2, 4
3. Target modules 消融: attn_only, qv_only, attn_ffn, all_linear

每组实验只跑有限步数（默认 500 步），快速对比趋势。
结果保存为 JSON 格式。

使用方式:
    cd p03_sft_finetuning

    # 运行全部消融实验
    python ablation_runner.py

    # 只跑 rank 消融
    python ablation_runner.py --ablation rank

    # 只跑 alpha 消融
    python ablation_runner.py --ablation alpha

    # 只跑 target_modules 消融
    python ablation_runner.py --ablation modules
"""

import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SFTConfig, LoRAConfig, AblationConfig, ablation_config


# ============================================================
# 1. 单次实验运行
# ============================================================
def run_single_experiment(
    experiment_name: str,
    cfg: SFTConfig,
    lcfg: LoRAConfig,
    max_steps: int = 500,
    eval_steps: int = 50,
) -> dict:
    """
    运行单次消融实验。
    
    Args:
        experiment_name: 实验名称
        cfg: SFT 配置
        lcfg: LoRA 配置
        max_steps: 最大训练步数
        eval_steps: 评估间隔
    
    Returns:
        实验结果字典
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
    )
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    from dataset import create_sft_dataset
    from shared.utils import set_seed
    
    set_seed(cfg.seed)
    
    print(f"\n{'─'*50}")
    print(f"  实验: {experiment_name}")
    print(f"  LoRA r={lcfg.lora_r}, alpha={lcfg.lora_alpha}")
    print(f"  Modules: {lcfg.target_modules}")
    print(f"{'─'*50}")
    
    start_time = time.time()
    
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    # 添加 LoRA
    peft_config = PeftLoraConfig(
        r=lcfg.lora_r,
        lora_alpha=lcfg.lora_alpha,
        lora_dropout=lcfg.lora_dropout,
        target_modules=lcfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    # 准备数据
    data_path = os.path.join(cfg.data_dir, "alpaca.jsonl")
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据不存在: {data_path}，跳过")
        return {"name": experiment_name, "error": "数据文件不存在"}
    
    dataset = create_sft_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        max_samples=2000,  # 消融实验用少量数据
    )
    
    # 划分训练/验证集
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
    
    from torch.utils.data import Subset
    train_ds = Subset(dataset, indices[:train_size].tolist())
    val_ds = Subset(dataset, indices[train_size:].tolist())
    
    # 训练
    output_dir = os.path.join("outputs/ablation", experiment_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",  # 消融不保存 checkpoint
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    
    # 训练并收集 loss
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    
    elapsed = time.time() - start_time
    
    # 收集日志中的 loss 曲线
    train_losses = []
    eval_losses = []
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"step": log.get("step", 0), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"step": log.get("step", 0), "loss": log["eval_loss"]})
    
    result = {
        "name": experiment_name,
        "lora_r": lcfg.lora_r,
        "lora_alpha": lcfg.lora_alpha,
        "target_modules": lcfg.target_modules,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_result.get("eval_loss", None),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "elapsed_seconds": elapsed,
        "max_steps": max_steps,
    }
    
    print(f"  完成! train_loss={result['final_train_loss']:.4f}, "
          f"eval_loss={result['final_eval_loss']:.4f}, "
          f"time={elapsed:.1f}s")
    
    # 释放显存
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


# ============================================================
# 2. LoRA Rank 消融
# ============================================================
def run_rank_ablation(ab_cfg: AblationConfig) -> list:
    """LoRA rank 消融实验"""
    print("\n" + "=" * 60)
    print("  消融实验 1: LoRA Rank")
    print(f"  测试 rank = {ab_cfg.lora_ranks}")
    print("=" * 60)
    
    results = []
    for rank in ab_cfg.lora_ranks:
        cfg = SFTConfig()
        lcfg = LoRAConfig()
        lcfg.lora_r = rank
        lcfg.lora_alpha = rank * 2  # 保持 alpha/r = 2
        
        result = run_single_experiment(
            experiment_name=f"rank_{rank}",
            cfg=cfg, lcfg=lcfg,
            max_steps=ab_cfg.max_steps,
            eval_steps=ab_cfg.eval_steps,
        )
        results.append(result)
    
    return results


# ============================================================
# 3. LoRA Alpha 消融
# ============================================================
def run_alpha_ablation(ab_cfg: AblationConfig) -> list:
    """LoRA alpha 消融实验"""
    print("\n" + "=" * 60)
    print("  消融实验 2: LoRA Alpha")
    print(f"  测试 alpha/r = {ab_cfg.alpha_ratios}")
    print("=" * 60)
    
    base_rank = 16
    results = []
    for ratio in ab_cfg.alpha_ratios:
        cfg = SFTConfig()
        lcfg = LoRAConfig()
        lcfg.lora_r = base_rank
        lcfg.lora_alpha = int(base_rank * ratio)
        
        result = run_single_experiment(
            experiment_name=f"alpha_r{base_rank}_a{lcfg.lora_alpha}",
            cfg=cfg, lcfg=lcfg,
            max_steps=ab_cfg.max_steps,
            eval_steps=ab_cfg.eval_steps,
        )
        results.append(result)
    
    return results


# ============================================================
# 4. Target Modules 消融
# ============================================================
def run_modules_ablation(ab_cfg: AblationConfig) -> list:
    """Target modules 消融实验"""
    print("\n" + "=" * 60)
    print("  消融实验 3: Target Modules")
    print("=" * 60)
    
    results = []
    for group_name, modules in ab_cfg.target_module_groups.items():
        cfg = SFTConfig()
        lcfg = LoRAConfig()
        lcfg.target_modules = modules
        
        result = run_single_experiment(
            experiment_name=f"modules_{group_name}",
            cfg=cfg, lcfg=lcfg,
            max_steps=ab_cfg.max_steps,
            eval_steps=ab_cfg.eval_steps,
        )
        results.append(result)
    
    return results


# ============================================================
# 5. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 SFT 消融实验")
    parser.add_argument("--ablation", type=str, default="all",
                       choices=["all", "rank", "alpha", "modules"],
                       help="要运行的消融实验类型")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="每组实验最大训练步数")
    parser.add_argument("--output", type=str, default=None,
                       help="结果保存路径")
    args = parser.parse_args()
    
    ab_cfg = ablation_config
    if args.max_steps:
        ab_cfg.max_steps = args.max_steps
    
    print("=" * 60)
    print("  p03 SFT 指令微调 - 消融实验")
    print("=" * 60)
    
    all_results = {}
    
    if args.ablation in ("all", "rank"):
        all_results["rank_ablation"] = run_rank_ablation(ab_cfg)
    
    if args.ablation in ("all", "alpha"):
        all_results["alpha_ablation"] = run_alpha_ablation(ab_cfg)
    
    if args.ablation in ("all", "modules"):
        all_results["modules_ablation"] = run_modules_ablation(ab_cfg)
    
    # 保存结果
    save_path = args.output or ab_cfg.save_results_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  ✅ 消融实验完成！")
    print(f"  结果已保存到: {save_path}")
    
    # 打印汇总
    for ablation_name, results in all_results.items():
        print(f"\n  📊 {ablation_name}:")
        for r in results:
            if "error" not in r:
                print(f"    {r['name']}: train={r['final_train_loss']:.4f}, "
                      f"eval={r['final_eval_loss']:.4f}, "
                      f"params={r['trainable_params']:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
