"""v07 SFT 推理: LoRA 权重合并 + 推理"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import SFTModel, inject_lora, LoRALinear
from shared.utils import get_device, load_checkpoint

def merge_lora_weights(model):
    """将所有 LoRA 权重合并到原始权重中（消除推理额外开销）。"""
    merged = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged += 1
    print(f"[LoRA] 已合并 {merged} 个 LoRA 权重")

@torch.no_grad()
def inference_demo(model, device):
    model.eval()
    img = torch.randn(1, 3, config.image_size, config.image_size).to(device)
    ids = torch.randint(4, config.vocab_size, (1, 16)).to(device)
    logits = model(img, ids)
    preds = logits.argmax(-1)
    print(f"Input: image + {ids.shape[1]} tokens")
    print(f"Output tokens: {preds[0].tolist()[:10]}...")

def main():
    device = get_device()
    model = SFTModel(config)
    model = inject_lora(model, r=config.lora_r, alpha=config.lora_alpha)
    ckpt = os.path.join(config.checkpoint_dir, "lora_model.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    model = model.to(device)
    
    print("=== LoRA 推理（未合并）===")
    inference_demo(model, device)
    
    print("\n=== LoRA 权重合并后推理 ===")
    merge_lora_weights(model)
    inference_demo(model, device)

if __name__ == "__main__":
    main()
