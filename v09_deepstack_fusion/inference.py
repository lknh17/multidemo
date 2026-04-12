"""v09 DeepStack 推理"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import DeepStackModel
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def demo(model, device):
    model.eval()
    img = torch.randn(4, 3, config.image_size, config.image_size).to(device)
    ids = torch.randint(4, config.vocab_size, (4, config.max_text_len)).to(device)
    out = model(img, ids)
    print(f"Fused image embedding: {out['fused_img'].shape}")
    print(f"Fused text embedding: {out['fused_txt'].shape}")
    print(f"Layer embeddings: {len(out['layer_img_embs'])} layers")
    sims = out['fused_img'] @ out['fused_txt'].t()
    print(f"Similarity matrix:\n{sims.cpu()}")

def main():
    device = get_device()
    model = DeepStackModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    demo(model, device)

if __name__ == "__main__":
    main()
