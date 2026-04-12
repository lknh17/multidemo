"""
V21 - 多模态数据工程 - 训练脚本
================================
支持三种训练模式：
1. augmented: 使用 CutMix / MixUp / RandAugment 增强
2. curriculum: 课程学习（先易后难）
3. balanced: 数据平衡（处理类别不平衡）

运行方式:
    cd v21_data_engineering
    python train.py --mode augmented
    python train.py --mode curriculum
    python train.py --mode balanced
"""

import os
import sys
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FullConfig
from model import SimplePatchClassifier, AugmentedTrainer, CurriculumScheduler
from dataset import create_dataloaders, SyntheticImageDataset
from data_ops import DataBalancer
from shared.utils import (
    set_seed, get_logger, save_checkpoint, count_parameters,
    get_device, AverageMeter, plot_training_curves, print_model_summary,
)


# ============================================================
# 1. 增强训练模式
# ============================================================
def train_augmented(config: FullConfig, device: torch.device, logger):
    """
    增强训练模式：在每个 batch 上随机应用 MixUp / CutMix / RandAugment。
    
    使用软标签（one-hot 混合），对应的 loss 需要用 soft cross-entropy。
    """
    logger.info("模式: Augmented Training (CutMix/MixUp/RandAugment)")
    
    train_loader, val_loader = create_dataloaders(config, imbalanced=False)
    model = SimplePatchClassifier(config).to(device)
    print_model_summary(model, "SimplePatchClassifier (Augmented)")
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    trainer = AugmentedTrainer(config)
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, config.num_epochs + 1):
        # ---- 训练 ----
        model.train()
        loss_meter = AverageMeter("train_loss")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # 应用随机增强
            aug_images, soft_labels = trainer.apply_augmentation(
                images, labels, aug_type="mixed"
            )
            soft_labels = soft_labels.to(device)
            
            logits = model(aug_images)
            
            # Soft Cross-Entropy: -Σ y_soft * log(softmax(logits))
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
        
        train_losses.append(loss_meter.avg)
        
        # ---- 验证 ----
        val_loss, val_acc = validate(model, val_loader, device, config)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {loss_meter.avg:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    # 保存结果
    save_checkpoint(
        model, optimizer, config.num_epochs, train_losses[-1],
        os.path.join(config.checkpoint_dir, "augmented_model.pt"),
    )
    plot_training_curves(
        train_losses, val_losses,
        val_metrics=val_accs, metric_name="Accuracy",
        save_path=os.path.join(config.log_dir, "augmented_curves.png"),
        title="Augmented Training",
    )
    
    return train_losses, val_losses, val_accs


# ============================================================
# 2. 课程学习训练模式
# ============================================================
def train_curriculum(config: FullConfig, device: torch.device, logger):
    """
    课程学习模式：先用简单样本训练，逐步引入困难样本。
    
    步骤：
    1. 用初始模型计算所有样本的难度
    2. 每个 epoch 按节奏函数 λ(t) 选择数据子集
    3. 在选中的子集上训练
    """
    logger.info("模式: Curriculum Learning")
    
    # 创建完整训练集
    full_dataset = SyntheticImageDataset(
        config, config.num_train_samples, imbalanced=False
    )
    _, val_loader = create_dataloaders(config, imbalanced=False)
    
    model = SimplePatchClassifier(config).to(device)
    print_model_summary(model, "SimplePatchClassifier (Curriculum)")
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = CurriculumScheduler(config, total_epochs=config.num_epochs)
    
    # 计算初始难度（使用未训练模型的 loss 作为难度）
    logger.info("计算样本难度...")
    all_images = []
    all_labels = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        all_images.append(sample["image"])
        all_labels.append(sample["label"])
    
    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)
    
    # 分批计算难度
    difficulties = []
    batch_size = 128
    for start in range(0, len(all_images), batch_size):
        end = min(start + batch_size, len(all_images))
        batch_diff = scheduler.compute_difficulty(
            model, all_images[start:end], all_labels[start:end], device
        )
        difficulties.append(batch_diff)
    difficulties = torch.cat(difficulties)
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, config.num_epochs + 1):
        # 选择本 epoch 的训练样本
        selected_indices = scheduler.select_samples(difficulties, epoch)
        fraction = scheduler.pacing_function(epoch)
        
        # 构建子集 DataLoader
        selected_images = all_images[selected_indices]
        selected_labels = all_labels[selected_indices]
        
        subset = torch.utils.data.TensorDataset(selected_images, selected_labels)
        subset_loader = torch.utils.data.DataLoader(
            subset, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        
        # 训练
        model.train()
        loss_meter = AverageMeter("train_loss")
        
        for images, labels in tqdm(subset_loader, desc=f"Epoch {epoch} ({fraction:.0%})", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
        
        train_losses.append(loss_meter.avg)
        
        # 更新难度（可选，每隔几个 epoch 重新评估）
        if epoch % 5 == 0:
            new_difficulties = []
            for start in range(0, len(all_images), batch_size):
                end = min(start + batch_size, len(all_images))
                batch_diff = scheduler.compute_difficulty(
                    model, all_images[start:end], all_labels[start:end], device
                )
                new_difficulties.append(batch_diff)
            difficulties = torch.cat(new_difficulties)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, device, config)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | Data: {len(selected_indices)}/{len(all_images)} ({fraction:.0%}) | "
            f"Train Loss: {loss_meter.avg:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    save_checkpoint(
        model, optimizer, config.num_epochs, train_losses[-1],
        os.path.join(config.checkpoint_dir, "curriculum_model.pt"),
    )
    plot_training_curves(
        train_losses, val_losses,
        val_metrics=val_accs, metric_name="Accuracy",
        save_path=os.path.join(config.log_dir, "curriculum_curves.png"),
        title="Curriculum Learning",
    )
    
    return train_losses, val_losses, val_accs


# ============================================================
# 3. 平衡训练模式
# ============================================================
def train_balanced(config: FullConfig, device: torch.device, logger):
    """
    平衡训练模式：在类别不平衡的数据上使用加权 loss + 过采样。
    """
    logger.info("模式: Balanced Training (Class Weighting + Oversampling)")
    
    train_loader, val_loader = create_dataloaders(config, imbalanced=True)
    
    # 获取训练集标签分布
    all_labels = train_loader.dataset.labels
    balancer = DataBalancer(num_classes=config.num_classes)
    
    # 计算类别权重
    class_weights = balancer.effective_number_weights(all_labels).to(device)
    logger.info(f"类别权重: {class_weights.tolist()}")
    
    # 打印类别分布
    from collections import Counter
    dist = Counter(all_labels)
    logger.info(f"类别分布: {dict(sorted(dist.items()))}")
    
    model = SimplePatchClassifier(config).to(device)
    print_model_summary(model, "SimplePatchClassifier (Balanced)")
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    
    # 使用加权 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        loss_meter = AverageMeter("train_loss")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
        
        train_losses.append(loss_meter.avg)
        
        val_loss, val_acc = validate(model, val_loader, device, config)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {loss_meter.avg:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    save_checkpoint(
        model, optimizer, config.num_epochs, train_losses[-1],
        os.path.join(config.checkpoint_dir, "balanced_model.pt"),
    )
    plot_training_curves(
        train_losses, val_losses,
        val_metrics=val_accs, metric_name="Accuracy",
        save_path=os.path.join(config.log_dir, "balanced_curves.png"),
        title="Balanced Training",
    )
    
    return train_losses, val_losses, val_accs


# ============================================================
# 验证函数
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    config: FullConfig,
) -> tuple:
    """验证并返回 (val_loss, val_accuracy)。"""
    model.eval()
    loss_meter = AverageMeter("val_loss")
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / max(total, 1)
    return loss_meter.avg, accuracy


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="V21 Data Engineering Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="augmented",
        choices=["augmented", "curriculum", "balanced"],
        help="训练模式: augmented / curriculum / balanced",
    )
    args = parser.parse_args()
    
    config = FullConfig()
    set_seed(config.seed)
    logger = get_logger("v21_train")
    device = get_device()
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("V21 多模态数据工程 - 训练")
    logger.info(f"模式: {args.mode}")
    logger.info("=" * 60)
    
    if args.mode == "augmented":
        train_augmented(config, device, logger)
    elif args.mode == "curriculum":
        train_curriculum(config, device, logger)
    elif args.mode == "balanced":
        train_balanced(config, device, logger)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
