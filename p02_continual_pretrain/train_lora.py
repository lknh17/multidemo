"""
p02 继续预训练 - LoRA 版本（对比实验）

用 LoRA 代替全参训练，相同数据、相同步数，验证：
1. LoRA 能否学到类似的中文风格
2. 权重改变量有多小
3. 是否缓解灾难遗忘

使用方式:
    cd p02_continual_pretrain
    python train_lora.py                           # 默认跑完整 1 epoch
    python train_lora.py --max-samples 5000        # 快速测试（约 10 分钟）
    python train_lora.py --max-steps 500           # 只跑 500 步
"""

import os
import sys
import argparse

os.environ["DS_BUILD_FUSED_ADAM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, lora_config
from dataset import create_pretrain_dataset


def train(args):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from shared.utils import set_seed

    cfg = config
    set_seed(cfg.seed)

    print("=" * 60)
    print("  p02 继续预训练 - LoRA 版本")
    print("=" * 60)

    # ---- Tokenizer ----
    print("\n[1/4] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base 模型 ----
    print("\n[2/4] 加载 Base 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ---- 套上 LoRA ----
    print("\n[2.5/4] 注入 LoRA 适配器...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # gradient checkpointing
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()  # LoRA + GC 必须加这行
        model.gradient_checkpointing_enable()
        print("  ✅ Gradient Checkpointing 已启用")

    # ---- 数据 ----
    print("\n[3/4] 准备数据...")
    data_path = args.data_path or "data/wiki_zh.jsonl"
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        return
    train_dataset = create_pretrain_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        max_samples=args.max_samples,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- 训练参数 ----
    print("\n[4/4] 开始训练（LoRA）...")
    lr = args.learning_rate or lora_config.learning_rate  # LoRA 默认 1e-4
    output_dir = args.output_dir or "outputs/pretrain_lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=lr,
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
        max_steps=args.max_steps if args.max_steps else -1,
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"\n  训练配置:")
    print(f"    数据集大小:     {len(train_dataset)}")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps}")
    print(f"    学习率:         {lr}")
    print(f"    LoRA rank:      {lora_config.lora_r}")
    print(f"    LoRA alpha:     {lora_config.lora_alpha}")
    print(f"    输出目录:       {output_dir}")

    trainer.train()

    # ---- 保存 LoRA 适配器 ----
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ---- 同时保存"合并后"的完整模型（方便对比）----
    print("\n合并 LoRA 权重到基座，保存完整模型...")
    merged_dir = os.path.join(output_dir, "final_merged")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print("\n" + "=" * 60)
    print("  ✅ LoRA 继续预训练完成！")
    print(f"  适配器: {final_dir}")
    print(f"  合并后: {merged_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
