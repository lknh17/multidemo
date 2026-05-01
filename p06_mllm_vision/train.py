"""
p06 MLLM 多模态视觉微调 - 主训练脚本

对 Qwen2.5-VL-3B 进行多模态指令微调，支持三种冻结策略：
1. freeze_vision: 冻结视觉编码器，只训练 LLM + 投影层（推荐）
2. partial_unfreeze: 冻结视觉编码器前 N 层，解冻后几层
3. full: 全模型端到端训练

支持功能:
1. LoRA 微调 LLM 部分
2. Gradient Checkpointing 显存优化
3. 多种冻结策略消融实验
4. 断点续训

使用方式:
    cd p06_mllm_vision
    # 基础训练（冻结 vision + LoRA on LLM）
    python train.py

    # 全参训练
    python train.py --freeze-strategy full --no-lora

    # 部分解冻视觉编码器
    python train.py --freeze-strategy partial_unfreeze --unfreeze-layers 4

    # 断点续训
    python train.py --resume-from-checkpoint outputs/mllm_vision/checkpoint-500
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MLLMConfig, LoRAConfig, ImageConfig, config, lora_config, image_config
from dataset import create_mllm_dataset, ImageProcessor


# ============================================================
# 1. 冻结策略
# ============================================================
def apply_freeze_strategy(model, strategy: str, unfreeze_layers: int = 4):
    """
    应用冻结策略。
    
    三种策略的显存和效果对比：
    - freeze_vision:     显存最低，训练最快，效果好
    - partial_unfreeze:  显存中等，让视觉特征适应任务
    - full:              显存最高，效果最好但容易过拟合
    """
    if strategy == "freeze_vision":
        # 冻结整个视觉编码器
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"  [冻结策略] freeze_vision: 冻结了 {frozen_count} 个视觉参数")
    
    elif strategy == "partial_unfreeze":
        # 先冻结所有视觉参数
        vision_params = []
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False
                vision_params.append((name, param))
        
        # 解冻最后 N 层
        unfrozen = 0
        for name, param in reversed(vision_params):
            if unfrozen < unfreeze_layers:
                if "layer" in name or "block" in name:
                    param.requires_grad = True
                    unfrozen += 1
        
        frozen = sum(1 for _, p in vision_params if not p.requires_grad)
        print(f"  [冻结策略] partial_unfreeze: 冻结 {frozen} 个, 解冻最后 {unfrozen} 层")
    
    elif strategy == "full":
        print("  [冻结策略] full: 全模型可训练")
    
    else:
        raise ValueError(f"未知冻结策略: {strategy}")
    
    return model


def apply_lora(model, lora_cfg: LoRAConfig):
    """
    在 LLM 部分应用 LoRA。
    
    LoRA 只作用于 LLM 的线性层，不影响视觉编码器。
    这样可以在冻结视觉编码器的同时，用极少参数微调 LLM。
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


# ============================================================
# 2. 训练主流程
# ============================================================
def train(cfg: MLLMConfig, lora_cfg: LoRAConfig, img_cfg: ImageConfig, args):
    """执行多模态微调"""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    
    # ---- 设置随机种子 ----
    from shared.utils import set_seed, print_model_summary
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p06 MLLM 多模态视觉微调")
    print("=" * 60)
    
    # ---- 加载 Processor 和 Tokenizer ----
    print("\n[1/5] 加载 Processor & Tokenizer...")
    
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code,
            min_pixels=img_cfg.min_pixels,
            max_pixels=img_cfg.max_pixels,
        )
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        processor = ImageProcessor(
            image_size=img_cfg.image_size,
            min_pixels=img_cfg.min_pixels,
            max_pixels=img_cfg.max_pixels,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 加载模型 ----
    print("\n[2/5] 加载模型...")
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    
    # ---- 应用冻结策略 ----
    print("\n[3/5] 应用冻结策略...")
    freeze_strategy = args.freeze_strategy or cfg.freeze_strategy
    model = apply_freeze_strategy(
        model, freeze_strategy,
        unfreeze_layers=args.unfreeze_layers or cfg.vision_unfreeze_layers,
    )
    
    # 应用 LoRA（可选）
    use_lora = lora_cfg.use_lora and not args.no_lora
    
    # 启用 gradient checkpointing（仅在不使用 LoRA 时，避免 FPE 问题）
    use_gc = cfg.gradient_checkpointing and not use_lora
    if use_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("  ✅ Gradient Checkpointing 已启用 (use_reentrant=False)")
    
    if use_lora:
        print("\n  应用 LoRA...")
        model = apply_lora(model, lora_cfg)
    
    print_model_summary(model, "Qwen2.5-VL-3B")
    
    # ---- 准备数据 ----
    print("\n[4/5] 准备数据...")
    data_path = args.data_path or os.path.join(cfg.data_dir, cfg.data_file)
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    train_dataset, val_dataset = create_mllm_dataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        max_samples=args.max_samples or cfg.max_samples,
        image_dir=args.image_dir or os.path.join(cfg.data_dir, "images"),
        val_ratio=cfg.val_ratio,
    )
    
    # ---- 配置训练参数 ----
    print("\n[5/5] 开始训练...")
    
    lr = lora_cfg.learning_rate if use_lora else (args.learning_rate or cfg.learning_rate)
    
    output_dir = args.output_dir or cfg.output_dir
    
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
        eval_steps=cfg.eval_steps if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # ---- 自定义 data collator ----
    def mllm_collate_fn(batch):
        """处理变长多模态输入的 collate 函数"""
        import torch
        
        # Pad input_ids, attention_mask, labels 到 batch 内最大长度
        max_len = max(b["input_ids"].shape[0] for b in batch)
        pad_token_id = tokenizer.pad_token_id or 0
        
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for b in batch:
            seq_len = b["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids_list.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
            attention_mask_list.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels_list.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        
        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
        
        # pixel_values: 拼接（不同图片 token 数不同）
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        
        return result
    
    # ---- 创建 Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=mllm_collate_fn,
    )
    
    # 打印训练信息
    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    total_steps = len(train_dataset) // eff_batch * cfg.num_train_epochs
    
    print(f"\n  训练配置:")
    print(f"    冻结策略:       {freeze_strategy}")
    print(f"    LoRA:           {'启用 (r={})'.format(lora_cfg.lora_r) if use_lora else '未使用'}")
    print(f"    数据集大小:     {len(train_dataset)}")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps} = {eff_batch}")
    print(f"    学习率:         {lr}")
    print(f"    预计步数:       ~{total_steps}")
    print(f"    输出目录:       {output_dir}")
    
    # ---- 训练 ----
    resume_from = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from)
    
    # ---- 保存最终模型 ----
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ MLLM 多模态微调完成！")
    print(f"  模型已保存到: {final_dir}")
    print("  下一步: python inference.py 测试多模态理解能力")
    print("=" * 60)


# ============================================================
# 3. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p06 MLLM 多模态视觉微调")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--freeze-strategy", type=str, default=None,
                       choices=["freeze_vision", "partial_unfreeze", "full"])
    parser.add_argument("--unfreeze-layers", type=int, default=None,
                       help="partial_unfreeze 时解冻的视觉层数")
    parser.add_argument("--no-lora", action="store_true",
                       help="不使用 LoRA，全参训练 LLM 部分")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（覆盖 config 中的 output_dir）")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    train(config, lora_config, image_config, args)


if __name__ == "__main__":
    main()
