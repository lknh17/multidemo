"""
p04 DPO 对齐训练 - 主训练脚本

使用 trl 库的 DPOTrainer / ORPOTrainer / KTOTrainer 进行偏好对齐训练。
支持 --algorithm 参数切换 DPO / SimPO / ORPO / KTO 四种算法。

支持功能:
1. DPO — 标准直接偏好优化
2. SimPO — 无需参考模型的简化 DPO
3. ORPO — Odds Ratio 偏好优化
4. KTO — Kahneman-Tversky 偏好优化
5. LoRA 高效微调
6. Beta 消融实验

使用方式:
    cd p04_dpo_alignment
    # DPO 训练
    python train.py --algorithm dpo

    # SimPO 训练
    python train.py --algorithm simpo

    # ORPO 训练
    python train.py --algorithm orpo

    # KTO 训练
    python train.py --algorithm kto

    # 指定 beta 值消融实验
    python train.py --algorithm dpo --beta 0.05

    # 使用自定义数据
    python train.py --algorithm dpo --data-path data/preference.jsonl
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DPOConfig, config, apply_algorithm_defaults


# ============================================================
# 1. 创建 LoRA 配置
# ============================================================
def create_lora_config(cfg: DPOConfig):
    """创建 LoRA 配置"""
    from peft import LoraConfig, TaskType
    
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return lora_config


# ============================================================
# 2. DPO / SimPO 训练
# ============================================================
def train_dpo(cfg: DPOConfig, args):
    """
    DPO / SimPO 训练。
    
    DPO 的核心思想：
    - 不需要训练单独的 reward model
    - 直接从偏好数据优化 policy model
    - Loss = -log σ(β · (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
    
    SimPO 的改进：
    - 不需要参考模型（ref-free）
    - 用序列平均 log-prob 替代 token-level log-prob
    - 引入 margin γ 增强区分度
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer
    from dataset import create_preference_dataset
    from shared.utils import set_seed
    
    set_seed(cfg.seed)
    
    algo_name = "SimPO" if cfg.algorithm == "simpo" else "DPO"
    print("=" * 60)
    print(f"  p04 对齐训练 - {algo_name}")
    print("=" * 60)
    
    # ---- 加载 Tokenizer ----
    print("\n[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- 加载 Policy 模型 ----
    print("\n[2/5] 加载 Policy 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  模型参数: {num_params:.2f}B")
    
    # ---- 参考模型 ----
    ref_model = None
    if cfg.algorithm == "dpo":
        # DPO 需要参考模型；SimPO 不需要
        if cfg.ref_model_name:
            print("\n  加载独立参考模型...")
            ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.ref_model_name,
                torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                trust_remote_code=True,
            )
        else:
            print("  参考模型: 使用 policy 初始权重（DPOTrainer 自动处理）")
    elif cfg.algorithm == "simpo":
        print("  SimPO: 无需参考模型")
    
    # ---- 准备数据 ----
    print("\n[3/5] 准备偏好数据...")
    data_path = args.data_path or cfg.data_path
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    dataset = create_preference_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        algorithm=cfg.algorithm,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_samples=args.max_samples,
    )
    
    # 转换为 HuggingFace Dataset（DPOTrainer 需要调用 .map()）
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_list(dataset.samples)
    
    # 划分训练/验证集
    split = hf_dataset.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    # ---- LoRA 配置 ----
    peft_config = None
    if cfg.use_lora:
        print("\n[4/5] 配置 LoRA...")
        peft_config = create_lora_config(cfg)
        trainable = cfg.lora_r * 2 * len(cfg.lora_target_modules)
        print(f"  LoRA r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    else:
        print("\n[4/5] 全参训练（不使用 LoRA）")
    
    # ---- 配置 DPO 训练参数 ----
    print("\n[5/5] 开始训练...")
    
    # 构建 DPO/SimPO 特有参数
    dpo_kwargs = {
        "beta": args.beta or cfg.beta,
        "loss_type": cfg.loss_type,
        "label_smoothing": cfg.label_smoothing,
    }
    
    if cfg.algorithm == "simpo":
        # SimPO 使用 simpo_gamma 和 cpo_alpha
        dpo_kwargs["simpo_gamma"] = cfg.simpo_gamma
        dpo_kwargs["cpo_alpha"] = cfg.cpo_alpha
        # SimPO 不需要参考模型，设置 ref_model=None
        dpo_kwargs["ref_model"] = None
    
    training_args = TRLDPOConfig(
        output_dir=os.path.join(cfg.output_dir, cfg.algorithm),
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
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        remove_unused_columns=False,
        **dpo_kwargs,
    )
    
    # ---- 创建 Trainer ----
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # 打印训练信息
    print(f"\n  训练配置:")
    print(f"    算法:           {algo_name}")
    print(f"    Beta:           {args.beta or cfg.beta}")
    print(f"    数据集大小:     {len(train_dataset)} (验证: {len(eval_dataset)})")
    print(f"    Batch Size:     {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps}")
    print(f"    学习率:         {args.learning_rate or cfg.learning_rate}")
    print(f"    LoRA:           {'是' if cfg.use_lora else '否'}")
    print(f"    输出目录:       {training_args.output_dir}")
    
    # ---- 训练 ----
    trainer.train()
    
    # ---- 保存 ----
    final_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 60)
    print(f"  ✅ {algo_name} 对齐训练完成！")
    print(f"  模型已保存到: {final_dir}")
    print("=" * 60)


# ============================================================
# 3. ORPO 训练
# ============================================================
def train_orpo(cfg: DPOConfig, args):
    """
    ORPO (Odds Ratio Preference Optimization) 训练。
    
    ORPO 的核心思想：
    - 不需要参考模型和额外的 SFT 阶段
    - 将偏好优化与语言建模统一为一个 loss
    - Loss = L_NLL + α · L_OR
    - L_OR 使用 odds ratio 度量 chosen vs rejected 的概率差异
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import ORPOConfig as TRLORPOConfig, ORPOTrainer
    from dataset import create_preference_dataset
    from shared.utils import set_seed
    
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p04 对齐训练 - ORPO")
    print("=" * 60)
    
    # 加载 Tokenizer & 模型
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    print(f"  ORPO 不需要参考模型，直接优化 policy")
    
    # 准备数据
    print("\n[2/4] 准备偏好数据...")
    data_path = args.data_path or cfg.data_path
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    dataset = create_preference_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        algorithm="orpo",
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_samples=args.max_samples,
    )
    
    eval_size = max(1, int(len(dataset) * cfg.eval_ratio))
    train_size = len(dataset) - eval_size
    
    import torch.utils.data
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    # LoRA
    peft_config = create_lora_config(cfg) if cfg.use_lora else None
    
    # 配置训练
    print("\n[3/4] 配置 ORPO 训练...")
    training_args = TRLORPOConfig(
        output_dir=os.path.join(cfg.output_dir, "orpo"),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.learning_rate or cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        beta=cfg.orpo_alpha,
        remove_unused_columns=False,
    )
    
    # 创建 Trainer
    print("\n[4/4] 开始 ORPO 训练...")
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"\n  训练配置:")
    print(f"    算法:         ORPO")
    print(f"    Alpha:        {cfg.orpo_alpha}")
    print(f"    数据集大小:   {len(train_dataset)}")
    print(f"    学习率:       {args.learning_rate or cfg.learning_rate}")
    
    trainer.train()
    
    final_dir = os.path.join(cfg.output_dir, "orpo", "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n  ✅ ORPO 训练完成！模型: {final_dir}")


# ============================================================
# 4. KTO 训练
# ============================================================
def train_kto(cfg: DPOConfig, args):
    """
    KTO (Kahneman-Tversky Optimization) 训练。
    
    KTO 的核心思想：
    - 基于 Kahneman-Tversky 前景理论
    - 不需要 chosen/rejected 配对，只需单条标注
    - 利用"损失厌恶"：人对负面结果的敏感度高于正面
    - 更适合实际场景（收集配对数据成本高）
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import KTOConfig as TRLKTOConfig, KTOTrainer
    from dataset import create_preference_dataset
    from shared.utils import set_seed
    
    set_seed(cfg.seed)
    
    print("=" * 60)
    print("  p04 对齐训练 - KTO")
    print("=" * 60)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    # 准备 KTO 格式数据
    print("\n[2/4] 准备 KTO 数据...")
    data_path = args.data_path or cfg.data_path
    
    if not os.path.exists(data_path):
        print(f"  ⚠️ 数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    dataset = create_preference_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        algorithm="kto",
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_samples=args.max_samples,
    )
    
    eval_size = max(1, int(len(dataset) * cfg.eval_ratio))
    train_size = len(dataset) - eval_size
    
    import torch.utils.data
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    # LoRA
    peft_config = create_lora_config(cfg) if cfg.use_lora else None
    
    # 配置 KTO 训练
    print("\n[3/4] 配置 KTO 训练...")
    training_args = TRLKTOConfig(
        output_dir=os.path.join(cfg.output_dir, "kto"),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=args.learning_rate or cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        beta=args.beta or cfg.beta,
        desirable_weight=cfg.kto_desirable_weight,
        undesirable_weight=cfg.kto_undesirable_weight,
        remove_unused_columns=False,
    )
    
    print("\n[4/4] 开始 KTO 训练...")
    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"\n  训练配置:")
    print(f"    算法:           KTO")
    print(f"    Beta:           {args.beta or cfg.beta}")
    print(f"    Desirable W:    {cfg.kto_desirable_weight}")
    print(f"    Undesirable W:  {cfg.kto_undesirable_weight}")
    print(f"    数据集大小:     {len(train_dataset)}")
    
    trainer.train()
    
    final_dir = os.path.join(cfg.output_dir, "kto", "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n  ✅ KTO 训练完成！模型: {final_dir}")


# ============================================================
# 5. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p04 DPO 对齐训练")
    parser.add_argument("--algorithm", type=str, default="dpo",
                       choices=["dpo", "simpo", "orpo", "kto"],
                       help="对齐算法: dpo/simpo/orpo/kto")
    parser.add_argument("--beta", type=float, default=None,
                       help="KL 惩罚系数 (DPO/SimPO/KTO)")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--no-lora", action="store_true",
                       help="关闭 LoRA，使用全参训练")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # 应用算法默认值
    cfg = DPOConfig()
    cfg.algorithm = args.algorithm
    cfg = apply_algorithm_defaults(cfg)
    
    if args.no_lora:
        cfg.use_lora = False
    
    # 分发到对应训练函数
    if args.algorithm in ("dpo", "simpo"):
        train_dpo(cfg, args)
    elif args.algorithm == "orpo":
        train_orpo(cfg, args)
    elif args.algorithm == "kto":
        train_kto(cfg, args)
    else:
        print(f"❌ 不支持的算法: {args.algorithm}")
        print(f"  支持: dpo, simpo, orpo, kto")


if __name__ == "__main__":
    main()
