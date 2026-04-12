"""v05 训练脚本"""
import os, sys
import torch, torch.nn as nn
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MultimodalQwen
from dataset import create_dataloaders, save_demo_data
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def main():
    set_seed(42); device = get_device(); logger = get_logger("v05")
    save_demo_data()
    train_loader, val_loader = create_dataloaders(config)
    model = MultimodalQwen(config).to(device)
    print_model_summary(model, "MultimodalQwen")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); lm = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(imgs, ids)
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); lm.update(loss.item())
        logger.info(f"Epoch {epoch} | Loss: {lm.avg:.4f}")
        if epoch == config.num_epochs:
            save_checkpoint(model, optimizer, epoch, lm.avg, os.path.join(config.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
