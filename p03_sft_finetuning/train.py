"""
p03 SFT 指令微调 - 主训练脚本

使用 HuggingFace Trainer + PEFT 进行 SFT 指令微调。
支持 4 种微调方法，通过 --method 参数切换：

1. full_finetune — 全参微调（更新所有参数）
2. lora — LoRA 微调（低秩适配）
3. qlora — QLoRA 微调（4-bit 量化 + LoRA）
4. dora — DoRA 微调（权重分解 + LoRA）

使用方式:
    cd p03_sft_finetuning

    # LoRA 微调（默认推荐）
    python train.py --method lora

    # QLoRA 微调（更省显存）
    python train.py --method qlora

    # DoRA 微调（效果更好）
    python train.py --method dora

    # 全参微调
    python train.py --method full_finetune

    # 使用 DeepSpeed
    deepspeed --num_gpus=1 train.py --method lora --deepspeed ds_config.json
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SFTConfig, LoRAConfig, create_method_config
from dataset import create_sft_dataset


# ============================================================
# 1. 模型加载（支持 4 种方法）
# ============================================================
def load_model_for_sft(cfg: SFTConfig, lcfg: LoRAConfig, args):
    """
    根据微调方法加载模型。
    
    - full_finetune: 直接加载，不添加 LoRA
    - lora: 加载 + 添加 LoRA adapter
    - qlora: 4-bit 量化加载 + 添加 LoRA adapter
    - dora: 加载 + 添加 DoRA adapter
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n  微调方法: {cfg.method}")
    
    # ---- 加载 Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 加载模型 ----
    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if args.flash_attn else "eager",
    }
    
    if cfg.method == "qlora":
        # QLoRA: 使用 bitsandbytes 4-bit 量化加载
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=lcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=lcfg.bnb_4bit_use_double_quant,
        )
        load_kwargs["quantization_config"] = quantization_config
        print("  ✅ 使用 4-bit 量化加载 (QLoRA)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if cfg.bf16 else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, **load_kwargs
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  模型参数: {num_params:.2f}B")
    
    # ---- 添加 LoRA / DoRA ----
    if cfg.method in ("lora", "qlora", "dora"):
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        
        # QLoRA 需要先准备模型
        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(model)
            print("  ✅ 已准备 4-bit 训练")
        
        # 创建 LoRA 配置
        peft_config = LoraConfig(
            r=lcfg.lora_r,
            lora_alpha=lcfg.lora_alpha,
            lora_dropout=lcfg.lora_dropout,
            target_modules=lcfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_dora=lcfg.use_dora if cfg.method == "dora" else False,
        )
        
        model = get_peft_model(model, peft_config)
        
        # 打印可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA rank: {lcfg.lora_r}, alpha: {lcfg.lora_alpha}")
        print(f"  Target modules: {lcfg.target_modules}")
        print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        if cfg.method == "dora":
            print("  ✅ DoRA 权重分解已启用")
    
    elif cfg.method == "full_finetune":
        print("  全参微调: 所有参数可训练")
    
    # 启用 gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("  ✅ Gradient Checkpointing 已启用")
    
    return model, tokenizer


# ============================================================
# 2. 训练主流程
# ============================================================
def train(cfg: SFTConfig, lcfg: LoRAConfig, args):
    """执行 SFT 指令微调"""
    from transformers import TrainingArguments, Trainer
    from shared.utils import set_seed
    
    set_seed(cfg.seed)
    
    print("=" * 60)
    print(f"  p03 SFT 指令微调 - {cfg.method}")
    print("=" * 60)
    
    # ---- 加载模型 ----
    print("\n[1/4] 加载模型...")
    model, tokenizer = load_model_for_sft(cfg, lcfg, args)
    
    # ---- 准备数据 ----
    print("\n[2/4] 准备数据...")
    data_path = args.data_path or os.path.join(cfg.data_dir, "alpaca.jsonl")
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    train_dataset = create_sft_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        data_format=args.data_format or cfg.dataset_format,
        max_samples=args.max_samples,
    )
    
    # 划分验证集
    val_dataset = None
    if cfg.val_ratio > 0:
        val_size = int(len(train_dataset) * cfg.val_ratio)
        train_size = len(train_dataset) - val_size
        
        import torch
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(cfg.seed))
        
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()
        
        # 创建子集
        from torch.utils.data import Subset
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
        
        print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # ---- 配置训练参数 ----
    print("\n[3/4] 配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy if val_dataset else "no",
        eval_steps=cfg.eval_steps if val_dataset else None,
        seed=cfg.seed,
        report_to="none",
        deepspeed=args.deepspeed,
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    # ---- 创建 Trainer ----
    print("\n[4/4] 开始训练...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # 打印训练信息
    total_steps = len(train_dataset) // (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    ) * cfg.num_train_epochs
    
    print(f"\n  训练配置:")
    print(f"    微调方法:       {cfg.method}")
    print(f"    数据集大小:     {len(train_dataset)}")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps} = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"    学习率:         {cfg.learning_rate}")
    print(f"    LR Scheduler:   {cfg.lr_scheduler_type}")
    print(f"    预计步数:       ~{total_steps}")
    print(f"    DeepSpeed:      {args.deepspeed or '未使用'}")
    print(f"    输出目录:       {cfg.output_dir}")
    
    # ---- 训练 ----
    resume_from = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from)
    
    # ---- 保存 ----
    final_dir = os.path.join(cfg.output_dir, "final")
    
    if cfg.method in ("lora", "qlora", "dora"):
        # LoRA 只保存 adapter 权重
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"  LoRA adapter 已保存到: {final_dir}")
    else:
        # 全参微调保存完整模型
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
    
    # 评估最终效果
    if val_dataset:
        eval_results = trainer.evaluate()
        print(f"\n  最终评估 loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    
    print("\n" + "=" * 60)
    print(f"  ✅ SFT {cfg.method} 微调完成！")
    print(f"  模型已保存到: {final_dir}")
    if cfg.method in ("lora", "qlora", "dora"):
        print(f"  合并权重: python merge_lora.py --adapter-path {final_dir}")
    print("  对比推理: python inference.py")
    print("=" * 60)


# ============================================================
# 3. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 SFT 指令微调")
    parser.add_argument("--method", type=str, default="lora",
                       choices=["full_finetune", "lora", "qlora", "dora"],
                       help="微调方法")
    parser.add_argument("--data-path", type=str, default=None,
                       help="数据文件路径")
    parser.add_argument("--data-format", type=str, default=None,
                       choices=["alpaca", "sharegpt"],
                       help="数据格式")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大训练样本数")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="学习率")
    parser.add_argument("--lora-r", type=int, default=None,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None,
                       help="LoRA alpha")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed 配置文件")
    parser.add_argument("--flash-attn", action="store_true",
                       help="使用 Flash Attention 2")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="从 checkpoint 恢复训练")
    parser.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed 需要
    args = parser.parse_args()
    
    # 根据 method 创建配置
    cfg, lcfg = create_method_config(args.method)
    
    # 命令行覆盖
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.lora_r:
        lcfg.lora_r = args.lora_r
    if args.lora_alpha:
        lcfg.lora_alpha = args.lora_alpha
    
    train(cfg, lcfg, args)


if __name__ == "__main__":
    main()
