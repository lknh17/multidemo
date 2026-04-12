"""v02 ViT 训练脚本"""
import os, sys, math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ViTConfig, config
from model import ViT
from dataset import create_dataloaders
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, plot_training_curves, print_model_summary, get_logger

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    loss_m, acc_m = AverageMeter("loss"), AverageMeter("acc")
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        acc = (logits.argmax(1) == labels).float().mean()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc.item(), images.size(0))
    scheduler.step()
    return loss_m.avg, acc_m.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m, acc_m = AverageMeter("loss"), AverageMeter("acc")
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc.item(), images.size(0))
    return loss_m.avg, acc_m.avg

def main():
    set_seed(42)
    logger = get_logger("v02_train")
    device = get_device()
    train_loader, val_loader = create_dataloaders(config)
    model = ViT(config).to(device)
    print_model_summary(model, "ViT")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0
    for epoch in range(1, config.num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc); val_accs.append(v_acc)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Train Loss:{t_loss:.4f} Acc:{t_acc:.4f} | Val Loss:{v_loss:.4f} Acc:{v_acc:.4f}")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(model, optimizer, epoch, t_loss, os.path.join(config.checkpoint_dir, "best_model.pt"), val_acc=v_acc)
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, "Accuracy",
                         os.path.join(config.log_dir, "training_curves.png"), "ViT CIFAR-10")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
