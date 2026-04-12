"""v03 Mini GPT 训练脚本"""
import os, sys, math
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MiniGPT
from dataset import create_dataloaders
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, plot_training_curves, print_model_summary
from tqdm import tqdm

def main():
    set_seed(42)
    logger = get_logger("v03_train")
    device = get_device()
    train_loader, val_loader, tokenizer = create_dataloaders(config)
    model = MiniGPT(config).to(device)
    print_model_summary(model, "MiniGPT")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses, val_losses = [], []
    best_val = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        loss_m = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(ids)
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_m.update(loss.item(), ids.size(0))
        train_losses.append(loss_m.avg)
        
        model.eval()
        val_m = AverageMeter("val")
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(ids)
                loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
                val_m.update(loss.item(), ids.size(0))
        val_losses.append(val_m.avg)
        ppl = math.exp(min(val_m.avg, 10))
        logger.info(f"Epoch {epoch} | Train Loss: {loss_m.avg:.4f} | Val Loss: {val_m.avg:.4f} | PPL: {ppl:.2f}")
        if val_m.avg < best_val:
            best_val = val_m.avg
            save_checkpoint(model, optimizer, epoch, loss_m.avg, os.path.join(config.checkpoint_dir, "best_model.pt"))
    
    plot_training_curves(train_losses, val_losses, save_path=os.path.join(config.log_dir, "curves.png"), title="MiniGPT")

if __name__ == "__main__":
    main()
