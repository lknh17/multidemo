"""v06 预训练模型评估: zero-shot 检索"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import PretrainModel
from dataset import create_dataloaders
from shared.utils import get_device, load_checkpoint
import torch.nn.functional as F

@torch.no_grad()
def zero_shot_retrieval(model, loader, device, k=5):
    model.eval()
    all_img, all_txt = [], []
    for batch in loader:
        vis = model.vision_enc(batch["image"].to(device))
        txt = model.text_enc(batch["input_ids"].to(device))
        all_img.append(F.normalize(model.img_proj(vis[:, 0]), dim=-1))
        all_txt.append(F.normalize(model.txt_proj(txt.mean(1)), dim=-1))
    all_img, all_txt = torch.cat(all_img), torch.cat(all_txt)
    sims = all_img @ all_txt.t()
    N = sims.size(0)
    labels = torch.arange(N, device=device)
    _, topk = sims.topk(k, dim=1)
    r_at_k = (topk == labels.unsqueeze(1)).any(1).float().mean().item()
    print(f"Zero-shot Retrieval R@{k}: {r_at_k:.4f} (N={N})")

def main():
    device = get_device()
    model = PretrainModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    _, val_loader = create_dataloaders(config)
    zero_shot_retrieval(model, val_loader, device)

if __name__ == "__main__":
    main()
