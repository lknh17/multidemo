"""v05 推理脚本: 展示多模态输入处理流程"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MultimodalQwen
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def demo_inference(model, device):
    model.eval()
    img = torch.randn(1, 3, config.image_size, config.image_size).to(device)
    prompt_ids = torch.randint(4, config.vocab_size, (1, 8)).to(device)
    print("=== 多模态前向传播流程 ===")
    vis_feat = model.vision_encoder(img)
    print(f"1. Vision Encoder: {img.shape} → {vis_feat.shape}")
    vis_tokens = model.resampler(vis_feat)
    print(f"2. Resampler: {vis_feat.shape} → {vis_tokens.shape} (压缩了 {vis_feat.shape[1]//vis_tokens.shape[1]}x)")
    txt_tokens = model.tok_embed(prompt_ids)
    print(f"3. Text Embedding: {prompt_ids.shape} → {txt_tokens.shape}")
    combined = torch.cat([vis_tokens, txt_tokens], dim=1)
    print(f"4. 拼接: vis[{vis_tokens.shape[1]}] + txt[{txt_tokens.shape[1]}] = [{combined.shape[1]}] tokens")
    logits = model(img, prompt_ids)
    print(f"5. LLM 输出: {logits.shape}")
    predicted = logits.argmax(-1)
    print(f"6. 预测 token ids: {predicted[0].tolist()[:10]}...")

def main():
    device = get_device()
    model = MultimodalQwen(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    demo_inference(model, device)

if __name__ == "__main__":
    main()
