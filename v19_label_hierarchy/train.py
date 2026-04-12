"""
V19 - 层级标签理解训练脚本
============================
支持三种模式：
1. hierarchical：层级分类（coarse→mid→fine）
2. multi_label：多标签分类（层级约束）
3. label_embed：标签嵌入（视觉-标签联合嵌入）

用法：
    python train.py --mode hierarchical
    python train.py --mode multi_label
    python train.py --mode label_embed
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

from config import LabelHierarchyFullConfig
from model import HierarchicalClassifier, ConstrainedMultiLabelModel, LabelEmbeddingModel
from dataset import create_hierarchical_dataloaders, create_multilabel_dataloaders, create_embedding_dataloaders


def train_hierarchical(config: LabelHierarchyFullConfig, logger):
    """训练层级分类器"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Hierarchical Classification Training (coarse→mid→fine)")
    logger.info("=" * 60)

    train_loader, val_loader = create_hierarchical_dataloaders(config)
    model = HierarchicalClassifier(config).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Tree structure: {config.hierarchy.num_labels_per_level}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meters = [AverageMeter() for _ in range(config.hierarchy.tree_depth)]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)

            # 各层目标标签（局部 ID）
            target_levels = [
                batch['coarse_label'].to(device),
                batch['mid_label'].to(device),
                batch['fine_label'].to(device),
            ]

            outputs = model(images, target_levels=target_levels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item())

            # 各层准确率
            for lv in range(config.hierarchy.tree_depth):
                pred = outputs['logits_per_level'][lv].argmax(-1)
                acc = (pred == target_levels[lv]).float().mean().item()
                acc_meters[lv].update(acc)

            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc_c': f"{acc_meters[0].avg:.2%}",
                'acc_f': f"{acc_meters[-1].avg:.2%}",
            })

        acc_str = ", ".join([f"L{i}={m.avg:.2%}" for i, m in enumerate(acc_meters)])
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, {acc_str}")

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "hierarchical_best.pt"))


def train_multi_label(config: LabelHierarchyFullConfig, logger):
    """训练多标签分类模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Multi-label Classification Training (hierarchy-constrained)")
    logger.info("=" * 60)

    train_loader, val_loader = create_multilabel_dataloaders(config)
    model = ConstrainedMultiLabelModel(config).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Total labels: {model.tree.total_labels}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        bce_meter = AverageMeter()
        consist_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            bce_meter.update(outputs['bce_loss'].item())
            consist_meter.update(outputs['consist_loss'].item())

            pbar.set_postfix({
                'bce': f"{bce_meter.avg:.4f}",
                'consist': f"{consist_meter.avg:.4f}",
            })

        logger.info(f"Epoch {epoch+1}: bce={bce_meter.avg:.4f}, consist={consist_meter.avg:.4f}")

        total_loss = bce_meter.avg + consist_meter.avg
        if total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "multilabel_best.pt"))


def train_label_embed(config: LabelHierarchyFullConfig, logger):
    """训练标签嵌入模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Label Embedding Training (visual-label joint embedding)")
    logger.info(f"Hyperbolic: {config.embedding.use_hyperbolic}")
    logger.info("=" * 60)

    train_loader, val_loader = create_embedding_dataloaders(config)
    model = LabelEmbeddingModel(config).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        dist_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            pos_labels = batch['pos_label'].to(device)
            neg_labels = batch['neg_labels'].to(device)

            outputs = model(images, pos_labels, neg_labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item())
            dist_meter.update(outputs['pos_distance'].mean().item())

            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'pos_dist': f"{dist_meter.avg:.3f}",
            })

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, pos_dist={dist_meter.avg:.3f}")

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "embedding_best.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="hierarchical",
                       choices=["hierarchical", "multi_label", "label_embed"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = LabelHierarchyFullConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    set_seed(config.seed)
    logger = get_logger("V19-LabelHierarchy")

    if args.mode == "hierarchical":
        train_hierarchical(config, logger)
    elif args.mode == "multi_label":
        train_multi_label(config, logger)
    elif args.mode == "label_embed":
        train_label_embed(config, logger)


if __name__ == "__main__":
    main()
