"""
V18 - 商品理解训练脚本
======================
python train.py --mode fine_grained
python train.py --mode attribute
python train.py --mode quality
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter
from config import ProductFullConfig
from fine_grained import MultiGranularityModel, ArcFaceHead
from model import ProductAttributeModel, QualityAssessmentModel
from dataset import create_fine_grained_dataloaders, create_attribute_dataloaders, create_quality_dataloaders


def train_fine_grained(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Fine-Grained Recognition Training")
    logger.info("=" * 60)
    train_loader, val_loader = create_fine_grained_dataloaders(config)
    model = MultiGranularityModel(config.fine_grained).to(device)
    arcface = ArcFaceHead(config.fine_grained.d_model, config.fine_grained.num_classes).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    params = list(model.parameters()) + list(arcface.parameters())
    optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        model.train(); loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            ce_loss = nn.functional.cross_entropy(outputs['fused_logits'], labels)
            global_loss = nn.functional.cross_entropy(outputs['global_logits'], labels)
            part_loss = sum(nn.functional.cross_entropy(pl, labels) for pl in outputs['part_logits']) / len(outputs['part_logits'])
            loss = ce_loss + 0.5 * global_loss + 0.3 * part_loss + 0.1 * outputs['diversity_loss']
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm); optimizer.step()
            loss_meter.update(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def train_attribute(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Product Attribute Extraction Training")
    logger.info("=" * 60)
    train_loader, _ = create_attribute_dataloaders(config)
    model = ProductAttributeModel(config.product_attr).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        model.train(); loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            outputs = model(images)
            cat_loss = nn.functional.cross_entropy(outputs['category'], batch['category'].to(device))
            brand_loss = nn.functional.cross_entropy(outputs['brand'], batch['brand'].to(device))
            color_loss = nn.functional.binary_cross_entropy(outputs['color'], batch['color'].to(device))
            loss = cat_loss + brand_loss + color_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm); optimizer.step()
            loss_meter.update(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def train_quality(config, logger):
    device = get_device()
    logger.info("=" * 60)
    logger.info("Quality Assessment Training")
    logger.info("=" * 60)
    train_loader, _ = create_quality_dataloaders(config)
    model = QualityAssessmentModel(config.quality).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        model.train(); loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            outputs = model(images)
            dim_loss = nn.functional.mse_loss(outputs['dim_scores'], batch['dim_scores'].to(device))
            overall_loss = nn.functional.mse_loss(outputs['overall_score'], batch['overall_score'].to(device))
            loss = dim_loss + overall_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm); optimizer.step()
            loss_meter.update(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fine_grained", choices=["fine_grained", "attribute", "quality"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = ProductFullConfig()
    if args.epochs: config.num_epochs = args.epochs
    set_seed(config.seed)
    logger = get_logger("V18-Product")
    {"fine_grained": train_fine_grained, "attribute": train_attribute, "quality": train_quality}[args.mode](config, logger)

if __name__ == "__main__":
    main()
