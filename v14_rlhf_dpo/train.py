"""
v14 RLHF / DPO 偏好对齐 - 训练脚本

支持多种训练模式:
- reward: 训练 Reward Model
- dpo: DPO 偏好对齐
- simpo: SimPO (无需参考模型)
- kto: KTO (无需偏好对)

用法:
    python train.py --mode dpo
    python train.py --mode reward
    python train.py --mode simpo
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter

from config import config
from model import PolicyModel, RewardModel, DPOTrainer
from dataset import create_preference_dataloader, create_kto_dataloader
from losses import (
    dpo_loss, simpo_loss, kto_loss, reward_model_loss, get_batch_logps
)


def train_dpo(config, device, logger):
    """DPO 训练"""
    logger.info("=" * 60)
    logger.info("DPO 偏好对齐训练")
    logger.info(f"  β = {config.dpo_beta}")
    logger.info(f"  label_smoothing = {config.label_smoothing}")
    logger.info("=" * 60)

    # 创建策略模型 + DPOTrainer (内含冻结的 ref 模型)
    policy = PolicyModel(config)
    trainer = DPOTrainer(policy, config, device)
    
    optimizer = torch.optim.AdamW(
        trainer.policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    dataloader = create_preference_dataloader(
        num_samples=1000, batch_size=config.batch_size,
        vocab_size=config.vocab_size,
    )
    
    best_accuracy = 0.0
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for epoch in range(config.num_epochs):
        trainer.policy.train()
        loss_meter.reset()
        acc_meter.reset()
        
        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            loss, metrics = trainer.compute_dpo_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.policy.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item())
            acc_meter.update(metrics["accuracy"])
            
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2%}",
                "margin": f"{metrics['reward_margin']:.3f}",
            })
        
        logger.info(
            f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, "
            f"accuracy={acc_meter.avg:.2%}, "
            f"margin={metrics['reward_margin']:.3f}"
        )
        
        if acc_meter.avg > best_accuracy:
            best_accuracy = acc_meter.avg
            save_checkpoint(
                trainer.policy, optimizer, epoch, loss_meter.avg,
                os.path.join(config.checkpoint_dir, "dpo_best.pt")
            )
    
    logger.info(f"DPO 训练完成！最佳准确率: {best_accuracy:.2%}")
    return trainer.policy


def train_simpo(config, device, logger):
    """SimPO 训练 (无需参考模型)"""
    logger.info("=" * 60)
    logger.info("SimPO 训练 (无需参考模型)")
    logger.info(f"  β = {config.simpo_beta}, γ = {config.simpo_gamma}")
    logger.info("=" * 60)

    policy = PolicyModel(config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    
    dataloader = create_preference_dataloader(
        num_samples=1000, batch_size=config.batch_size, vocab_size=config.vocab_size,
    )
    
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        policy.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        
        pbar = tqdm(dataloader, desc=f"SimPO Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # SimPO 使用 average_log_prob
            chosen_logps = policy.get_logps(chosen_ids, chosen_labels, average=True)
            rejected_logps = policy.get_logps(rejected_ids, rejected_labels, average=True)
            
            loss, metrics = simpo_loss(
                chosen_logps, rejected_logps,
                beta=config.simpo_beta, gamma=config.simpo_gamma,
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item())
            acc_meter.update(metrics["accuracy"])
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2%}"})
        
        logger.info(f"SimPO Epoch {epoch+1}: loss={loss_meter.avg:.4f}, acc={acc_meter.avg:.2%}")
        
        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            save_checkpoint(policy, optimizer, epoch, loss_meter.avg,
                          os.path.join(config.checkpoint_dir, "simpo_best.pt"))
    
    logger.info(f"SimPO 训练完成！最佳准确率: {best_acc:.2%}")


def train_reward_model(config, device, logger):
    """训练 Reward Model"""
    logger.info("=" * 60)
    logger.info("Reward Model 训练 (Bradley-Terry)")
    logger.info("=" * 60)

    model = RewardModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate * 5, weight_decay=config.weight_decay,
    )
    
    dataloader = create_preference_dataloader(
        num_samples=1000, batch_size=config.batch_size, vocab_size=config.vocab_size,
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        
        pbar = tqdm(dataloader, desc=f"RM Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            
            chosen_rewards = model(chosen_ids)
            rejected_rewards = model(rejected_ids)
            
            loss, metrics = reward_model_loss(chosen_rewards, rejected_rewards)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item())
            acc_meter.update(metrics["accuracy"])
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2%}"})
        
        logger.info(f"RM Epoch {epoch+1}: loss={loss_meter.avg:.4f}, acc={acc_meter.avg:.2%}")


def train_kto(config, device, logger):
    """KTO 训练"""
    logger.info("=" * 60)
    logger.info("KTO 训练 (无需偏好对)")
    logger.info(f"  β = {config.kto_beta}")
    logger.info("=" * 60)

    import copy
    policy = PolicyModel(config).to(device)
    ref_model = copy.deepcopy(policy).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    
    dataloader = create_kto_dataloader(
        num_samples=1000, batch_size=config.batch_size, vocab_size=config.vocab_size,
    )
    
    for epoch in range(config.num_epochs):
        policy.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(dataloader, desc=f"KTO Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            is_desirable = batch["is_desirable"].to(device)
            
            policy_logps = policy.get_logps(input_ids, labels)
            with torch.no_grad():
                ref_logps = ref_model.get_logps(input_ids, labels)
            
            loss, metrics = kto_loss(
                policy_logps, ref_logps, is_desirable,
                beta=config.kto_beta,
                desirable_weight=config.kto_desirable_weight,
                undesirable_weight=config.kto_undesirable_weight,
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item())
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        
        logger.info(f"KTO Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="v14 偏好对齐训练")
    parser.add_argument("--mode", type=str, default="dpo",
                       choices=["dpo", "simpo", "reward", "kto"],
                       help="训练模式")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger = get_logger("v14_rlhf_dpo")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    logger.info(f"设备: {device}")
    logger.info(f"训练模式: {args.mode}")
    
    if args.mode == "dpo":
        train_dpo(config, device, logger)
    elif args.mode == "simpo":
        train_simpo(config, device, logger)
    elif args.mode == "reward":
        train_reward_model(config, device, logger)
    elif args.mode == "kto":
        train_kto(config, device, logger)


if __name__ == "__main__":
    main()
