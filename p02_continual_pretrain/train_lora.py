"""
p02 继续预训练 - LoRA 预训练对比脚本

用 LoRA 进行继续预训练，与全参预训练对比：
- 可训练参数: 全参 100% vs LoRA ~2-5%
- 显存: LoRA 大幅降低（不需要全参梯度和完整优化器状态）
- 效果: 全参通常更好，但 LoRA 性价比高

使用方式:
    cd p02_continual_pretrain
    python train_lora.py
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PretrainConfig, LoRAPretrainConfig, config, lora_config
from dataset import create_pretrain_dataset


def train_lora(cfg: PretrainConfig, lora_cfg: LoRAPretrainConfig, args):
    """LoRA 继续预训练"""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from shared.utils import set_seed, print_model_summary
    
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p02 继续预训练 - LoRA 模式")
    print("=" * 60)
    
    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 模型 ----
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    # ---- LoRA 配置 ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    print_model_summary(model, "LoRA Pretrain Model")
    
    # ---- 数据 ----
    data_path = args.data_path or "data/wiki_zh.jsonl"
    if not os.path.exists(data_path):
        print(f"  ⚠️ 请先运行: python download_data.py")
        return
    
    train_dataset = create_pretrain_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        max_samples=args.max_samples,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # ---- 训练 ----
    output_dir = os.path.join(cfg.output_dir, "lora")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=lora_cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # 保存 LoRA 权重
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    print("\n" + "=" * 60)
    print("  ✅ LoRA 预训练完成！")
    print(f"  LoRA 权重已保存到: {output_dir}/final")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="p02 LoRA 预训练")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    train_lora(config, lora_config, args)


if __name__ == "__main__":
    main()
