"""v08 相关性推理"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import DualEncoder
from dataset import create_dataloaders
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def relevance_eval(model, loader, device):
    model.eval()
    all_img, all_txt, all_labels = [], [], []
    for batch in loader:
        ie, te, _ = model(batch["image"].to(device), batch["input_ids"].to(device))
        all_img.append(ie); all_txt.append(te); all_labels.append(batch["label"])
    all_img, all_txt = torch.cat(all_img), torch.cat(all_txt)
    labels = torch.cat(all_labels)
    sims = all_img @ all_txt.t()
    N = sims.size(0)
    gt = torch.arange(N, device=device)
    for k in [1, 5, 10]:
        _, topk = sims.topk(k, dim=1)
        recall = (topk == gt.unsqueeze(1)).any(1).float().mean().item()
        print(f"  R@{k}: {recall:.4f}")

def main():
    device = get_device()
    _, val_loader = create_dataloaders(config)
    for name in ["InfoNCE", "CircleLoss"]:
        ckpt = os.path.join(config.checkpoint_dir, f"{name}_best.pt")
        model = DualEncoder(config).to(device)
        if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
        print(f"\n=== {name} ===")
        relevance_eval(model, val_loader, device)

if __name__ == "__main__":
    main()
