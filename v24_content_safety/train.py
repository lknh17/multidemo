"""
V24 - 内容安全训练脚本
======================
python train.py --mode safety_cls
python train.py --mode adversarial
python train.py --mode watermark
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter
from config import SafetyFullConfig
from safety_modules import ContentClassifier, WatermarkEmbedder, AdversarialAttacker
from model import RobustClassifier
from dataset import create_safety_dataloaders, create_adversarial_dataloaders, create_watermark_dataloaders


def train_safety_cls(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Safety Classification Training")
    logger.info("=" * 60)
    train_loader, val_loader = create_safety_dataloaders(config)
    model = ContentClassifier(config.safety_cls).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            loss = criterion(outputs['logits'], labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())

        # 验证
        model.eval()
        val_loss_meter = AverageMeter()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images)
                val_loss = criterion(outputs['logits'], labels)
                val_loss_meter.update(val_loss.item())
                preds = outputs['predictions']
                correct += (preds == labels).all(dim=-1).sum().item()
                total += labels.shape[0]
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, val_loss={val_loss_meter.avg:.4f}, exact_match={correct/max(total,1):.4f}")


def train_adversarial(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Adversarial Robust Training")
    logger.info("=" * 60)
    train_loader, val_loader = create_adversarial_dataloaders(config)
    model = RobustClassifier(config).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.classifier.parameters()):,}")
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images, labels)
            if 'adv_logits' in outputs:
                loss = model.adversarial_loss(outputs['clean_logits'], outputs['adv_logits'], labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(outputs['logits'], labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def train_watermark(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Watermark Embedding Training")
    logger.info("=" * 60)
    train_loader, val_loader = create_watermark_dataloaders(config)
    model = WatermarkEmbedder(config.watermark).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            watermark = batch['watermark'].to(device)
            outputs = model(images, watermark)
            # 水印检测损失 + 图像质量损失
            detect_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs['detected_logits'], (watermark > 0).float()
            )
            quality_loss = outputs['embed_mse']
            loss = detect_loss + 10.0 * quality_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())
            acc_meter.update(outputs['bit_accuracy'].item())

        # 验证
        model.eval()
        val_acc = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                watermark = batch['watermark'].to(device)
                outputs = model(images, watermark)
                val_acc.update(outputs['bit_accuracy'].item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, train_bit_acc={acc_meter.avg:.4f}, val_bit_acc={val_acc.avg:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="safety_cls", choices=["safety_cls", "adversarial", "watermark"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = SafetyFullConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    set_seed(config.seed)
    logger = get_logger("V24-Safety")
    {"safety_cls": train_safety_cls, "adversarial": train_adversarial, "watermark": train_watermark}[args.mode](config, logger)

if __name__ == "__main__":
    main()
