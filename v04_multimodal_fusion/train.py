"""v04 CLIP 训练脚本"""
import os, sys
import torch, torch.nn as nn
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MiniCLIP, InfoNCELoss
from dataset import create_dataloaders, save_demo_data
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, plot_training_curves, print_model_summary

def compute_recall_at_k(img_emb, txt_emb, k=5):
    sims = img_emb @ txt_emb.t()
    N = sims.size(0)
    _, topk = sims.topk(k, dim=1)
    labels = torch.arange(N, device=sims.device).unsqueeze(1)
    return (topk == labels).any(dim=1).float().mean().item()

def main():
    set_seed(42); device = get_device(); logger = get_logger("v04")
    save_demo_data()
    train_loader, val_loader = create_dataloaders(config)
    model = MiniCLIP(config).to(device)
    print_model_summary(model, "MiniCLIP")
    criterion = InfoNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    train_losses, val_losses = [], []
    best = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        model.train(); lm = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, ids = batch["image"].to(device), batch["input_ids"].to(device)
            img_e, txt_e, scale = model(imgs, ids)
            loss = criterion(img_e, txt_e, scale)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lm.update(loss.item(), imgs.size(0))
        train_losses.append(lm.avg)
        
        model.eval(); vm = AverageMeter("val"); all_img, all_txt = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs, ids = batch["image"].to(device), batch["input_ids"].to(device)
                ie, te, s = model(imgs, ids)
                vm.update(criterion(ie, te, s).item(), imgs.size(0))
                all_img.append(ie); all_txt.append(te)
        val_losses.append(vm.avg)
        all_img, all_txt = torch.cat(all_img), torch.cat(all_txt)
        r5 = compute_recall_at_k(all_img, all_txt, k=5)
        logger.info(f"Epoch {epoch} | Train: {lm.avg:.4f} | Val: {vm.avg:.4f} | R@5: {r5:.4f}")
        if vm.avg < best:
            best = vm.avg
            save_checkpoint(model, optimizer, epoch, lm.avg, os.path.join(config.checkpoint_dir, "best.pt"))
    
    plot_training_curves(train_losses, val_losses, save_path=os.path.join(config.log_dir, "curves.png"), title="MiniCLIP")

if __name__ == "__main__":
    main()
