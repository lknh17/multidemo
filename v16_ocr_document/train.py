"""
V16 - OCR / 文档理解 / 广告文字提取训练脚本
============================================
支持三种模式：
1. ocr：文字检测 + 识别训练
2. document：LayoutLM 文档理解训练
3. ad_text：广告文字提取训练

用法：
    python train.py --mode ocr
    python train.py --mode document
    python train.py --mode ad_text
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

from config import OCRDocumentFullConfig
from ocr_modules import TextDetector
from model import DocumentUnderstandingModel, AdTextExtractionModel
from dataset import create_ocr_dataloaders, create_document_dataloaders, create_ad_text_dataloaders


def train_ocr(config: OCRDocumentFullConfig, logger):
    """训练 OCR 文字检测"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("OCR Text Detection Training (DBNet)")
    logger.info("=" * 60)

    train_loader, val_loader = create_ocr_dataloaders(config)
    model = TextDetector(config.ocr_det).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            gt_prob = batch['prob_map'].to(device)
            gt_thresh = batch['thresh_map'].to(device)

            outputs = model(images)
            losses = model.compute_loss(outputs, gt_prob, gt_thresh)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss_meter.update(losses['total'].item())
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "ocr_best.pt"))


def train_document(config: OCRDocumentFullConfig, logger):
    """训练文档理解模型"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Document Understanding Training (LayoutLM-style)")
    logger.info("=" * 60)

    # 使用较短序列长度加速
    config.document.max_text_len = 128
    train_loader, val_loader = create_document_dataloaders(config)
    model = DocumentUnderstandingModel(config.document).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        cls_meter = AverageMeter()
        tok_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            token_ids = batch['token_ids'].to(device)
            bboxes = batch['bboxes'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            token_labels = batch['token_labels'].to(device)
            mask = batch['attention_mask'].to(device)

            outputs = model(token_ids, bboxes, images)

            # 文档分类损失
            cls_loss = nn.functional.cross_entropy(outputs['cls_logits'], labels)

            # 序列标注损失（只算有效 token）
            tok_logits = outputs['token_logits'].reshape(-1, config.document.num_labels)
            tok_targets = token_labels.reshape(-1)
            tok_loss = nn.functional.cross_entropy(tok_logits, tok_targets, ignore_index=0)

            loss = cls_loss + tok_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            cls_meter.update(cls_loss.item())
            tok_meter.update(tok_loss.item())
            pbar.set_postfix({'cls': f"{cls_meter.avg:.4f}", 'tok': f"{tok_meter.avg:.4f}"})

        total = cls_meter.avg + tok_meter.avg
        logger.info(f"Epoch {epoch+1}: cls={cls_meter.avg:.4f}, tok={tok_meter.avg:.4f}")
        if total < best_loss:
            best_loss = total
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(os.path.dirname(__file__), "checkpoints", "document_best.pt"))


def train_ad_text(config: OCRDocumentFullConfig, logger):
    """训练广告文字提取"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Ad Text Extraction Training")
    logger.info("=" * 60)

    train_loader, val_loader = create_ad_text_dataloaders(config)
    model = AdTextExtractionModel(config).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 只训练检测 + 分类头
    det_params = list(model.detector.parameters())
    cls_params = list(model.type_classifier.parameters()) + \
                 list(model.relation_layers.parameters()) + \
                 list(model.spatial_encoder.parameters()) + \
                 list(model.region_proj.parameters())

    optimizer = optim.AdamW(
        [{'params': det_params, 'lr': config.learning_rate},
         {'params': cls_params, 'lr': config.learning_rate * 2}],
        weight_decay=config.weight_decay
    )

    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            bboxes = batch['bboxes'].to(device)
            types = batch['types'].to(device)
            mask = batch['region_mask'].to(device)
            region_images = batch['region_images'].to(device)

            outputs = model(images, region_bboxes=bboxes, region_images=region_images)

            # 检测损失（简化：prob map）
            # 实际场景中应该从 bboxes 生成 GT prob_map
            det_loss = outputs['det']['prob_map'].mean() * 0.01  # 占位

            # 类型分类损失
            if 'type_logits' in outputs:
                type_logits = outputs['type_logits']  # [B, N_reg, num_types]
                valid = mask.bool()
                if valid.any():
                    cls_loss = nn.functional.cross_entropy(
                        type_logits[valid], types[valid],
                        ignore_index=-1,
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=device)
            else:
                cls_loss = torch.tensor(0.0, device=device)

            loss = det_loss + cls_loss

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
                           os.path.join(os.path.dirname(__file__), "checkpoints", "ad_text_best.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ocr", choices=["ocr", "document", "ad_text"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = OCRDocumentFullConfig()
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size

    set_seed(config.seed)
    logger = get_logger("V16-OCR-Document")

    if args.mode == "ocr":
        train_ocr(config, logger)
    elif args.mode == "document":
        train_document(config, logger)
    elif args.mode == "ad_text":
        train_ad_text(config, logger)


if __name__ == "__main__":
    main()
