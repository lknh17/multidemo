"""
v14 RLHF / DPO 偏好对齐 - 推理与对比

对比不同对齐方法的效果：
1. 基础 SFT 模型 vs DPO 对齐模型
2. 不同 β 值对 DPO 的影响
3. DPO vs SimPO vs KTO 的 loss 曲线对比
"""
import os
import sys
import copy

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_device, get_logger

from config import config
from model import PolicyModel, DPOTrainer
from dataset import create_preference_dataloader
from losses import dpo_loss, simpo_loss


def demo_dpo_beta_analysis(device, logger):
    """分析 β 值对 DPO 的影响"""
    logger.info("=" * 60)
    logger.info("实验 1: DPO β 值敏感性分析")
    logger.info("=" * 60)
    
    betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # 模拟不同的 log-ratio 值
    chosen_logps = torch.randn(100)
    rejected_logps = chosen_logps - torch.abs(torch.randn(100)) * 0.5
    ref_chosen_logps = torch.zeros(100)
    ref_rejected_logps = torch.zeros(100)
    
    logger.info(f"{'β':>6} | {'Loss':>8} | {'Accuracy':>10} | {'Margin':>8}")
    logger.info("-" * 45)
    
    for beta in betas:
        loss, metrics = dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta,
        )
        logger.info(
            f"{beta:6.2f} | {loss.item():8.4f} | {metrics['accuracy']:10.2%} | "
            f"{metrics['reward_margin']:8.4f}"
        )
    
    logger.info("\n结论：β 越大 → loss 越大, 偏离 ref 越少（保守）")
    logger.info("      β 越小 → loss 越小, 偏好对齐越激进")


def demo_simpo_vs_dpo(device, logger):
    """SimPO vs DPO 对比"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 2: SimPO vs DPO 对比")
    logger.info("=" * 60)
    
    # 模拟数据
    n = 200
    chosen_logps = -torch.abs(torch.randn(n)) * 2
    rejected_logps = chosen_logps - torch.abs(torch.randn(n)) * 0.3
    ref_chosen = torch.zeros(n)
    ref_rejected = torch.zeros(n)
    
    # DPO loss
    dpo_l, dpo_m = dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, beta=0.1)
    
    # SimPO loss (使用平均概率模拟)
    simpo_l, simpo_m = simpo_loss(chosen_logps / 20, rejected_logps / 20, beta=2.0, gamma=0.5)
    
    logger.info(f"DPO:   loss={dpo_l.item():.4f}, accuracy={dpo_m['accuracy']:.2%}")
    logger.info(f"SimPO: loss={simpo_l.item():.4f}, accuracy={simpo_m['accuracy']:.2%}")
    logger.info("\nSimPO 优势：无需参考模型，节省 50% 显存")


def demo_reward_margin_evolution(device, logger):
    """模拟 DPO 训练过程中 reward margin 的变化"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 3: DPO 训练 — Reward Margin 演进")
    logger.info("=" * 60)
    
    set_seed(42)
    policy = PolicyModel(config).to(device)
    trainer = DPOTrainer(policy, config, device)
    
    optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=1e-4)
    dataloader = create_preference_dataloader(
        num_samples=200, batch_size=8, vocab_size=config.vocab_size,
    )
    
    logger.info(f"{'Step':>5} | {'Loss':>8} | {'Accuracy':>10} | {'Margin':>8}")
    logger.info("-" * 45)
    
    step = 0
    trainer.policy.train()
    for batch in dataloader:
        loss, metrics = trainer.compute_dpo_loss(batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), 1.0)
        optimizer.step()
        
        if step % 5 == 0:
            logger.info(
                f"{step:5d} | {loss.item():8.4f} | {metrics['accuracy']:10.2%} | "
                f"{metrics['reward_margin']:8.4f}"
            )
        step += 1
        if step >= 30:
            break
    
    logger.info("\n训练过程中 margin 逐渐增大 → 模型学会区分 chosen/rejected")


def demo_label_smoothing_effect(device, logger):
    """Label Smoothing 对 DPO 的影响"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 4: DPO Label Smoothing 效果")
    logger.info("=" * 60)
    
    # 模拟含噪声的偏好数据（10% 标注错误）
    n = 500
    chosen_logps = -torch.abs(torch.randn(n)) * 2
    rejected_logps = chosen_logps.clone()
    
    # 90% 正确偏好，10% 错误偏好
    correct_mask = torch.rand(n) > 0.1
    rejected_logps[correct_mask] -= torch.abs(torch.randn(correct_mask.sum())) * 0.5
    rejected_logps[~correct_mask] += torch.abs(torch.randn((~correct_mask).sum())) * 0.5
    
    ref_c, ref_r = torch.zeros(n), torch.zeros(n)
    
    for ls in [0.0, 0.05, 0.1, 0.2]:
        loss, metrics = dpo_loss(chosen_logps, rejected_logps, ref_c, ref_r,
                                beta=0.1, label_smoothing=ls)
        logger.info(f"label_smoothing={ls:.2f}: loss={loss.item():.4f}, "
                    f"acc={metrics['accuracy']:.2%}")
    
    logger.info("\n适度的 label smoothing (0.05-0.1) 在有噪声标注时更鲁棒")


def main():
    set_seed(42)
    device = get_device()
    logger = get_logger("v14_inference")
    
    logger.info(f"设备: {device}")
    
    demo_dpo_beta_analysis(device, logger)
    demo_simpo_vs_dpo(device, logger)
    demo_reward_margin_evolution(device, logger)
    demo_label_smoothing_effect(device, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("全部推理实验完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
