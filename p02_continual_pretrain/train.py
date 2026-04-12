"""
p02 继续预训练 - 主训练脚本（全参 + DeepSpeed）

使用 HuggingFace Trainer + DeepSpeed ZeRO 进行继续预训练。
目标：让模型学习更多中文知识，提升中文生成质量。

支持功能:
1. 全参预训练（默认）
2. DeepSpeed ZeRO-2/3 显存优化
3. Gradient Checkpointing
4. 断点续训
5. 多种学习率策略

使用方式:
    cd p02_continual_pretrain
    # 基础训练
    python train.py

    # 使用 DeepSpeed ZeRO-2
    deepspeed --num_gpus=1 train.py --deepspeed ds_config_zero2.json

    # 指定学习率策略
    python train.py --lr-scheduler cosine --learning-rate 2e-5

    # 断点续训
    python train.py --resume-from-checkpoint outputs/pretrain/checkpoint-500
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PretrainConfig, config
from dataset import create_pretrain_dataset


# ============================================================
# 1. 训练主流程
# ============================================================
def train(cfg: PretrainConfig, args):
    """执行继续预训练"""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    
    # ---- 设置随机种子 ----
    from shared.utils import set_seed
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p02 继续预训练 - 全参训练")
    print("=" * 60)
    
    # ---- 加载 Tokenizer ----
    print("\n[1/4] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 加载模型 ----
    print("\n[2/4] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  模型参数: {num_params:.2f}B")
    
    # 启用 gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  ✅ Gradient Checkpointing 已启用")
    
    # ---- 准备数据 ----
    print("\n[3/4] 准备数据...")
    data_path = args.data_path or "data/wiki_zh.jsonl"
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    train_dataset = create_pretrain_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        max_samples=args.max_samples,
    )
    
    # 数据 collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # CLM（因果语言模型），不是 MLM（掩码语言模型）
    )
    
    # ---- 配置训练参数 ----
    print("\n[4/4] 开始训练...")
    
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.learning_rate or cfg.learning_rate,
        lr_scheduler_type=args.lr_scheduler or cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",  # 不使用 wandb
        deepspeed=args.deepspeed,
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    # ---- 创建 Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 打印训练信息
    total_steps = len(train_dataset) // (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    ) * cfg.num_train_epochs
    
    print(f"\n  训练配置:")
    print(f"    数据集大小:     {len(train_dataset)}")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps} = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"    学习率:         {args.learning_rate or cfg.learning_rate}")
    print(f"    LR Scheduler:   {args.lr_scheduler or cfg.lr_scheduler_type}")
    print(f"    预计步数:       ~{total_steps}")
    print(f"    DeepSpeed:      {args.deepspeed or '未使用'}")
    print(f"    输出目录:       {cfg.output_dir}")
    
    # ---- 训练 ----
    resume_from = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from)
    
    # ---- 保存最终模型 ----
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ 继续预训练完成！")
    print(f"  模型已保存到: {final_dir}")
    print("  下一步: python inference.py 对比训练前后效果")
    print("=" * 60)


# ============================================================
# 2. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p02 继续预训练")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-scheduler", type=str, default=None,
                       choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed 需要
    args = parser.parse_args()
    
    train(config, args)


if __name__ == "__main__":
    main()
