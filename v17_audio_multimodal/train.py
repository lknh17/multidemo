"""
V17 - 音频理解 / 全模态训练脚本
=================================
支持三种模式：
1. audio_encoder：AST 音频编码器训练
2. clap：CLAP 音频-文本对比学习
3. omni_modal：全模态融合训练

用法：
    python train.py --mode audio_encoder
    python train.py --mode clap
    python train.py --mode omni_modal
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

from config import AudioMultimodalFullConfig
from audio_modules import AudioSpectrogramTransformer, AudioEventDetector
from model import CLAPModel, OmniModalModel
from dataset import create_audio_text_dataloaders, create_omni_modal_dataloaders


def train_audio_encoder(config: AudioMultimodalFullConfig, logger):
    """训练 AST 音频编码器（分类任务）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("AST Audio Encoder Training (Classification)")
    logger.info("=" * 60)

    train_loader, val_loader = create_audio_text_dataloaders(config)

    encoder = AudioSpectrogramTransformer(config.audio_enc).to(device)
    classifier = nn.Linear(config.audio_enc.d_model, 5).to(device)  # 5 类音频
    logger.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        encoder.train()
        classifier.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            mel = batch['mel_spec'].to(device)
            labels = batch['audio_type'].to(device)

            feats = encoder(mel)
            logits = classifier(feats)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item())
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(encoder, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "ast_best.pt"))


def train_clap(config: AudioMultimodalFullConfig, logger):
    """训练 CLAP 音频-文本对齐"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("CLAP Audio-Text Contrastive Training")
    logger.info("=" * 60)

    train_loader, val_loader = create_audio_text_dataloaders(config)
    model = CLAPModel(config.clap, config.audio_enc).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            mel = batch['mel_spec'].to(device)
            tokens = batch['token_ids'].to(device)

            outputs = model(mel, tokens)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item())
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}",
                             'temp': f"{outputs['temperature'].item():.2f}"})

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "clap_best.pt"))


def train_omni_modal(config: AudioMultimodalFullConfig, logger):
    """训练全模态融合模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Omni-Modal Fusion Training")
    logger.info("=" * 60)

    train_loader, val_loader = create_omni_modal_dataloaders(config)
    model = OmniModalModel(config.omni, config.audio_enc).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            tokens = batch['token_ids'].to(device)
            mel = batch['mel_spec'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, tokens, mel)
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
                           os.path.join(os.path.dirname(__file__), "checkpoints", "omni_best.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="clap",
                       choices=["audio_encoder", "clap", "omni_modal"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = AudioMultimodalFullConfig()
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size

    set_seed(config.seed)
    logger = get_logger("V17-Audio-Multimodal")

    if args.mode == "audio_encoder":
        train_audio_encoder(config, logger)
    elif args.mode == "clap":
        train_clap(config, logger)
    elif args.mode == "omni_modal":
        train_omni_modal(config, logger)


if __name__ == "__main__":
    main()
