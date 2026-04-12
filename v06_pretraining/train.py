"""v06 预训练脚本"""
import os, sys, torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import PretrainModel, PretrainLoss
from dataset import create_dataloaders
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def main():
    set_seed(42); device = get_device(); logger = get_logger("v06")
    train_loader, val_loader = create_dataloaders(config)
    model = PretrainModel(config).to(device)
    print_model_summary(model, "PretrainModel")
    loss_fn = PretrainLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); meters = {k: AverageMeter(k) for k in ["itc", "itm", "cap", "total"]}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            itm_labels = batch["itm_label"].to(device)
            cap_labels = batch["cap_labels"].to(device)
            outputs = model(imgs, ids)
            loss, details = loss_fn(outputs, itm_labels, cap_labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            for k, v in details.items(): meters[k].update(v)
        
        info = " | ".join(f"{k}: {m.avg:.4f}" for k, m in meters.items())
        logger.info(f"Epoch {epoch} | {info}")
    save_checkpoint(model, optimizer, config.num_epochs, meters["total"].avg, os.path.join(config.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
