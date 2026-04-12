"""v10 概念预测推理"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from config import config
from model import ConceptModel
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def predict_concepts(model, device):
    model.eval()
    img = torch.randn(2, 3, config.image_size, config.image_size).to(device)
    ids = torch.randint(4, config.vocab_size, (2, config.max_text_len)).to(device)
    out = model(img, ids)
    for key in ["industry", "brand", "intent"]:
        preds = out["img_concepts"][key].argmax(-1)
        print(f"  {key}: {preds.tolist()}")
    attr_preds = (out["img_concepts"]["attributes"].sigmoid() > 0.5).int()
    print(f"  attributes: {attr_preds.tolist()}")

def main():
    device = get_device()
    model = ConceptModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    print("=== 商业概念预测 ===")
    predict_concepts(model, device)

if __name__ == "__main__":
    main()
