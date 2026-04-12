"""
V20 - 知识增强多模态嵌入训练脚本
=================================
支持三种模式：
1. kg_embed：TransE/TransR 知识图谱嵌入训练
2. kg_visual：KG 增强视觉模型训练
3. distill：从 KG-enhanced 教师蒸馏到纯视觉学生

用法：
    python train.py --mode kg_embed
    python train.py --mode kg_visual
    python train.py --mode distill
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter

from config import KnowledgeEmbeddingFullConfig
from kg_modules import TransEEmbedding, TransREmbedding
from model import KGEnhancedVisualModel, KnowledgeDistillModel
from dataset import create_kg_dataloaders, create_image_entity_dataloaders, create_distill_dataloaders


def train_kg_embed(config: KnowledgeEmbeddingFullConfig, logger):
    """训练知识图谱嵌入（TransE + TransR）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Knowledge Graph Embedding Training (TransE & TransR)")
    logger.info("=" * 60)

    train_loader, val_loader = create_kg_dataloaders(config)

    # TransE
    transe = TransEEmbedding(config.kg).to(device)
    logger.info(f"TransE parameters: {sum(p.numel() for p in transe.parameters()):,}")

    # TransR
    transr = TransREmbedding(config.kg).to(device)
    logger.info(f"TransR parameters: {sum(p.numel() for p in transr.parameters()):,}")

    opt_e = optim.Adam(transe.parameters(), lr=config.learning_rate * 5)
    opt_r = optim.Adam(transr.parameters(), lr=config.learning_rate * 5)

    best_loss_e = float('inf')
    best_loss_r = float('inf')

    for epoch in range(config.num_epochs):
        transe.train()
        transr.train()
        loss_e_meter = AverageMeter()
        loss_r_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            pos_h = batch['pos_h'].to(device)
            pos_r = batch['pos_r'].to(device)
            pos_t = batch['pos_t'].to(device)
            neg_h = batch['neg_h'].to(device)
            neg_r = batch['neg_r'].to(device)
            neg_t = batch['neg_t'].to(device)

            # TransE
            loss_e = transe.compute_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            opt_e.zero_grad()
            loss_e.backward()
            opt_e.step()

            # TransR
            loss_r = transr.compute_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            opt_r.zero_grad()
            loss_r.backward()
            opt_r.step()

            loss_e_meter.update(loss_e.item())
            loss_r_meter.update(loss_r.item())
            pbar.set_postfix({
                'TransE': f"{loss_e_meter.avg:.4f}",
                'TransR': f"{loss_r_meter.avg:.4f}",
            })

        logger.info(f"Epoch {epoch+1}: TransE={loss_e_meter.avg:.4f}, TransR={loss_r_meter.avg:.4f}")

        if loss_e_meter.avg < best_loss_e:
            best_loss_e = loss_e_meter.avg
            save_checkpoint(transe, opt_e, epoch, best_loss_e,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "transe_best.pt"))

        if loss_r_meter.avg < best_loss_r:
            best_loss_r = loss_r_meter.avg
            save_checkpoint(transr, opt_r, epoch, best_loss_r,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "transr_best.pt"))


def train_kg_visual(config: KnowledgeEmbeddingFullConfig, logger):
    """训练 KG 增强视觉模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("KG-Enhanced Visual Model Training")
    logger.info("=" * 60)

    train_loader, val_loader = create_image_entity_dataloaders(config)
    model = KGEnhancedVisualModel(config).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, entity_ids)
            loss = nn.functional.cross_entropy(outputs['logits'], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item())
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "kg_visual_best.pt"))


def train_distill(config: KnowledgeEmbeddingFullConfig, logger):
    """训练知识蒸馏模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Training (KG → Visual)")
    logger.info("=" * 60)

    train_loader, val_loader = create_distill_dataloaders(config)
    model = KnowledgeDistillModel(config).to(device)

    teacher_params = sum(p.numel() for p in model.teacher.parameters())
    student_params = sum(p.numel() for p in model.student.parameters())
    logger.info(f"Teacher parameters: {teacher_params:,} (frozen)")
    logger.info(f"Student parameters: {student_params:,} (trainable)")

    # 只训练学生和对齐投影
    trainable = list(model.student.parameters()) + list(model.align_proj.parameters())
    optimizer = optim.AdamW(trainable, lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        task_meter = AverageMeter()
        distill_meter = AverageMeter()
        feat_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, entity_ids, labels)

            optimizer.zero_grad()
            outputs['loss'].backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()

            task_meter.update(outputs['task_loss'].item())
            distill_meter.update(outputs['distill_loss'].item())
            feat_meter.update(outputs['feat_loss'].item())
            pbar.set_postfix({
                'task': f"{task_meter.avg:.4f}",
                'distill': f"{distill_meter.avg:.4f}",
                'feat': f"{feat_meter.avg:.4f}",
            })

        logger.info(f"Epoch {epoch+1}: task={task_meter.avg:.4f}, "
                    f"distill={distill_meter.avg:.4f}, feat={feat_meter.avg:.4f}")
        total = task_meter.avg + distill_meter.avg
        if total < best_loss:
            best_loss = total
            save_checkpoint(model.student, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "distill_best.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="kg_embed",
                       choices=["kg_embed", "kg_visual", "distill"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = KnowledgeEmbeddingFullConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    set_seed(config.seed)
    logger = get_logger("V20-Knowledge-Embedding")

    if args.mode == "kg_embed":
        train_kg_embed(config, logger)
    elif args.mode == "kg_visual":
        train_kg_visual(config, logger)
    elif args.mode == "distill":
        train_distill(config, logger)


if __name__ == "__main__":
    main()
