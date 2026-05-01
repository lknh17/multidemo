#!/usr/bin/env python3
"""
p06 三种冻结策略对比实验
直接使用 Trainer 运行，避免 train.py 中的兼容性问题
"""
import os, sys, json, time
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# Workaround: 禁用 cuBLAS 的 cublasLt 避免 FPE (divide error in libcublasLt.so)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    TrainingArguments, Trainer
)

# 禁用 TF32 避免 cuBLAS divide error
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from peft import LoraConfig, get_peft_model, TaskType
from dataset import create_mllm_dataset

SAMPLES = 200
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

EXPERIMENTS = [
    {
        "name": "exp1_freeze_vision_lora",
        "freeze": "freeze_vision",
        "use_lora": True,
        "lr": 2e-4,
    },
    {
        "name": "exp2_partial_unfreeze_lora",
        "freeze": "partial_unfreeze",
        "use_lora": True,
        "lr": 2e-4,
    },
    {
        "name": "exp3_full_no_lora",
        "freeze": "full",
        "use_lora": False,
        "lr": 1e-5,
    },
]


def collate_fn(batch):
    pad_id = tokenizer.pad_token_id or 0
    max_len = max(b["input_ids"].shape[0] for b in batch)
    ids, masks, labs = [], [], []
    for b in batch:
        sl = b["input_ids"].shape[0]
        pl = max_len - sl
        ids.append(torch.cat([b["input_ids"], torch.full((pl,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([b["attention_mask"], torch.zeros(pl, dtype=torch.long)]))
        labs.append(torch.cat([b["labels"], torch.full((pl,), -100, dtype=torch.long)]))
    r = {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks), "labels": torch.stack(labs)}
    if "pixel_values" in batch[0]:
        r["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        r["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return r


def run_experiment(exp_cfg):
    name = exp_cfg["name"]
    output_dir = f"outputs/{name}"
    
    print(f"\n{'='*60}")
    print(f"  实验: {name}")
    print(f"  冻结策略: {exp_cfg['freeze']}")
    print(f"  LoRA: {exp_cfg['use_lora']}")
    print(f"  学习率: {exp_cfg['lr']}")
    print(f"{'='*60}")
    
    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager",
    )
    
    # 冻结策略
    if exp_cfg["freeze"] == "freeze_vision":
        for n, p in model.named_parameters():
            if "visual" in n:
                p.requires_grad = False
        print("  冻结了视觉编码器")
    elif exp_cfg["freeze"] == "partial_unfreeze":
        vision_params = []
        for n, p in model.named_parameters():
            if "visual" in n:
                p.requires_grad = False
                vision_params.append((n, p))
        # 解冻最后 4 层
        unfrozen = 0
        for n, p in reversed(vision_params):
            if unfrozen < 4 and ("layer" in n or "block" in n):
                p.requires_grad = True
                unfrozen += 1
        print(f"  部分解冻视觉编码器 (最后 {unfrozen} 层)")
    else:
        print("  全模型可训练")
    
    # LoRA
    if exp_cfg["use_lora"]:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 全参训练时启用 gradient checkpointing 节省显存
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    
    # 训练
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=exp_cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=False,
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=not exp_cfg["use_lora"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if not exp_cfg["use_lora"] else None,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    
    # 保存
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # 记录结果
    result = {
        "name": name,
        "freeze": exp_cfg["freeze"],
        "use_lora": exp_cfg["use_lora"],
        "lr": exp_cfg["lr"],
        "train_time_sec": round(elapsed, 1),
        "train_loss": round(trainer.state.log_history[-1].get("train_loss", -1), 4),
    }
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    result["trainable_params_M"] = round(trainable / 1e6, 1)
    result["trainable_pct"] = round(trainable / total * 100, 2)
    
    print(f"\n  ✅ {name} 完成!")
    print(f"  训练时间: {elapsed:.1f}s")
    print(f"  最终 loss: {result['train_loss']}")
    print(f"  可训练参数: {result['trainable_params_M']}M ({result['trainable_pct']}%)")
    
    # 释放显存
    del model, trainer
    torch.cuda.empty_cache()
    
    return result


if __name__ == "__main__":
    # 加载 processor 和数据（只加载一次）
    print("加载 Processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, min_pixels=256*28*28, max_pixels=512*28*28)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("加载数据...")
    train_dataset, _ = create_mllm_dataset(
        "data/llava_instruct_20k.json", processor, tokenizer,
        max_seq_length=512, max_samples=SAMPLES,
        image_dir="data/images", val_ratio=0.0,
    )
    
    os.makedirs("outputs", exist_ok=True)
    
    # 运行实验
    results = []
    for exp in EXPERIMENTS:
        try:
            r = run_experiment(exp)
            results.append(r)
        except Exception as e:
            print(f"\n  ❌ {exp['name']} 失败: {e}")
            results.append({"name": exp["name"], "error": str(e)})
    
    # 对比汇总
    print(f"\n\n{'='*70}")
    print("  三种冻结策略对比结果")
    print(f"{'='*70}")
    print(f"{'实验':<35} {'可训练参数':<15} {'Train Loss':<12} {'训练时间':<10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} {'ERROR':<15} {r['error'][:30]}")
        else:
            params = f"{r['trainable_params_M']}M ({r['trainable_pct']}%)"
            print(f"{r['name']:<35} {params:<15} {r['train_loss']:<12} {r['train_time_sec']}s")
    print("=" * 70)
    
    # 保存结果
    with open("outputs/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 outputs/comparison_results.json")
