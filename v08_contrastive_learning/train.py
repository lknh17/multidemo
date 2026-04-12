"""v08 对比学习训练: 多种 Loss 对比实验"""
import os, sys, torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import DualEncoder
from losses import InfoNCELoss, CircleLoss
from dataset import create_dataloaders, save_demo_data
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def train_with_loss(loss_name, loss_fn, config, device, logger):
    train_loader, val_loader = create_dataloaders(config)
    model = DualEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); lm = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"[{loss_name}] E{epoch}", leave=False):
            ie, te, s = model(batch["image"].to(device), batch["input_ids"].to(device))
            if loss_name == "InfoNCE":
                loss = loss_fn(ie, te, s)
            else:
                loss = loss_fn(ie, te)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lm.update(loss.item())
        logger.info(f"[{loss_name}] Epoch {epoch} | Loss: {lm.avg:.4f}")
    save_checkpoint(model, optimizer, config.num_epochs, lm.avg, os.path.join(config.checkpoint_dir, f"{loss_name}_best.pt"))
    return model

def main():
    set_seed(42); device = get_device(); logger = get_logger("v08")
    save_demo_data()
    
    losses = {"InfoNCE": InfoNCELoss(config.temperature), "CircleLoss": CircleLoss()}
    for name, loss_fn in losses.items():
        logger.info(f"\n{'='*40} Training with {name} {'='*40}")
        train_with_loss(name, loss_fn, config, device, logger)

if __name__ == "__main__":
    main()
