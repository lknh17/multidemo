"""
V15 - 视频理解与 Dense Captioning 训练脚本
==========================================
支持三种训练模式：
1. dense_caption：端到端 Dense Video Captioning
2. temporal_grounding：时序 Grounding（Moment Retrieval）
3. video_encoder：单独训练视频编码器（时序分类）

用法：
    python train.py --mode dense_caption
    python train.py --mode temporal_grounding
    python train.py --mode video_encoder
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

from config import VideoDenseCaptionFullConfig
from model import DenseVideoCaptioningModel, TemporalGroundingModel
from video_encoder import VideoEncoder
from dataset import (
    create_dense_caption_dataloaders,
    create_grounding_dataloaders,
    DenseCaptionDataset,
)


def train_dense_caption(config: VideoDenseCaptionFullConfig, logger):
    """训练 Dense Video Captioning 模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Dense Video Captioning Training")
    logger.info("=" * 60)

    # 数据
    train_loader, val_loader = create_dense_caption_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} videos, Val: {len(val_loader.dataset)} videos")

    # 模型
    model = DenseVideoCaptioningModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_meter = AverageMeter()
        reg_meter = AverageMeter()
        cap_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in pbar:
            video = batch['video'].to(device)
            spans = batch['spans'].to(device)
            captions = batch['captions'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(video, caption_tokens=captions, gt_spans=spans, gt_labels=labels)
            losses = model.compute_loss(outputs, spans, labels, captions)

            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # 记录
            loss_meter.update(losses['total'].item())
            cls_meter.update(losses['cls_loss'].item())
            reg_meter.update(losses['reg_loss'].item())
            if 'caption_loss' in losses:
                cap_meter.update(losses['caption_loss'].item())

            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'cls': f"{cls_meter.avg:.4f}",
                'reg': f"{reg_meter.avg:.4f}",
                'cap': f"{cap_meter.avg:.4f}",
            })

        # 验证
        val_loss = validate_dense_caption(model, val_loader, device)
        logger.info(
            f"Epoch {epoch + 1}: train_loss={loss_meter.avg:.4f}, "
            f"val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "dense_caption_best.pt"))
            logger.info(f"  → Best model saved (val_loss={val_loss:.4f})")


def validate_dense_caption(model, val_loader, device):
    """验证 Dense Caption 模型"""
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            video = batch['video'].to(device)
            spans = batch['spans'].to(device)
            captions = batch['captions'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(video, caption_tokens=captions, gt_spans=spans, gt_labels=labels)
            losses = model.compute_loss(outputs, spans, labels, captions)
            loss_meter.update(losses['total'].item())

    return loss_meter.avg


def train_temporal_grounding(config: VideoDenseCaptionFullConfig, logger):
    """训练 Temporal Grounding 模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Temporal Grounding Training")
    logger.info("=" * 60)

    train_loader, val_loader = create_grounding_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = TemporalGroundingModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * config.num_epochs
    )

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in pbar:
            video = batch['video'].to(device)
            query = batch['query'].to(device)
            query_mask = batch['query_mask'].to(device)
            span = batch['span'].to(device)

            outputs = model(video, query, query_mask)

            # 构建 GT 格式
            gt_spans = span.unsqueeze(1)  # [B, 1, 2]
            gt_labels = torch.ones(span.shape[0], 1, device=device, dtype=torch.long)

            losses = model.compute_loss(outputs, gt_spans, gt_labels)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_meter.update(losses['total'].item())
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

        logger.info(f"Epoch {epoch + 1}: loss={loss_meter.avg:.4f}")

        if loss_meter.avg < best_val_loss:
            best_val_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, loss_meter.avg,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "grounding_best.pt"))


def train_video_encoder(config: VideoDenseCaptionFullConfig, logger):
    """单独训练视频编码器（时序活动分类）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Video Encoder Training (Activity Classification)")
    logger.info("=" * 60)

    # 简单的分类任务：判断视频中有多少个事件
    from torch.utils.data import DataLoader

    train_ds = DenseCaptionDataset(config, split="train")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)

    encoder = VideoEncoder(config.video).to(device)
    classifier = nn.Linear(config.video.d_model, 6).to(device)  # 0~5 个事件

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=config.learning_rate)

    logger.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"Temporal model: {config.video.temporal_model}")

    for epoch in range(min(config.num_epochs, 10)):
        encoder.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            video = batch['video'].to(device)
            n_events = batch['n_events'].to(device)  # [B]

            # 编码 + 分类
            temporal_feat = encoder(video)  # [B, T, D]
            pooled = temporal_feat.mean(dim=1)  # [B, D]
            logits = classifier(pooled)  # [B, 6]

            loss = nn.functional.cross_entropy(logits, n_events.clamp(max=5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(-1) == n_events.clamp(max=5)).float().mean()
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())

            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{acc_meter.avg:.2%}",
            })

        logger.info(
            f"Epoch {epoch + 1}: loss={loss_meter.avg:.4f}, acc={acc_meter.avg:.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description="V15 Video Dense Caption Training")
    parser.add_argument("--mode", type=str, default="dense_caption",
                       choices=["dense_caption", "temporal_grounding", "video_encoder"],
                       help="Training mode")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temporal_model", type=str, default="timesformer",
                       choices=["timesformer", "video_swin", "conv3d"])
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_size", type=int, default=64,
                       help="Frame resolution (use 64 for quick demo)")
    args = parser.parse_args()

    config = VideoDenseCaptionFullConfig()

    # 覆盖配置
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    config.video.temporal_model = args.temporal_model
    config.video.num_frames = args.num_frames
    config.video.frame_size = args.frame_size

    set_seed(config.seed)
    logger = get_logger("V15-VideoDenseCaption")

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Temporal model: {config.video.temporal_model}")
    logger.info(f"Frames: {config.video.num_frames}, Resolution: {config.video.frame_size}")

    if args.mode == "dense_caption":
        train_dense_caption(config, logger)
    elif args.mode == "temporal_grounding":
        train_temporal_grounding(config, logger)
    elif args.mode == "video_encoder":
        train_video_encoder(config, logger)


if __name__ == "__main__":
    main()
