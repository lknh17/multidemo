"""v04 图文检索推理"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MiniCLIP
from dataset import create_dataloaders
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def retrieval_demo(model, val_loader, device, top_k=5):
    model.eval()
    all_img, all_txt, all_labels = [], [], []
    for batch in val_loader:
        ie, te, _ = model(batch["image"].to(device), batch["input_ids"].to(device))
        all_img.append(ie); all_txt.append(te); all_labels.append(batch["label"])
    all_img, all_txt = torch.cat(all_img), torch.cat(all_txt)
    all_labels = torch.cat(all_labels)
    
    # 图搜文 (Image → Text)
    sims = all_img @ all_txt.t()
    print("=== Image → Text Retrieval ===")
    for i in range(5):
        _, topk = sims[i].topk(top_k)
        q_label = all_labels[i].item()
        r_labels = [all_labels[j].item() for j in topk]
        hit = q_label in r_labels
        print(f"  Query(class={q_label}) → Top-{top_k} results: {r_labels} {'✅' if hit else '❌'}")
    
    # 文搜图 (Text → Image)
    sims_t2i = all_txt @ all_img.t()
    print("\n=== Text → Image Retrieval ===")
    for i in range(5):
        _, topk = sims_t2i[i].topk(top_k)
        q_label = all_labels[i].item()
        r_labels = [all_labels[j].item() for j in topk]
        hit = q_label in r_labels
        print(f"  Query(class={q_label}) → Top-{top_k} results: {r_labels} {'✅' if hit else '❌'}")

def main():
    device = get_device()
    model = MiniCLIP(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    _, val_loader = create_dataloaders(config)
    retrieval_demo(model, val_loader, device)

if __name__ == "__main__":
    main()
