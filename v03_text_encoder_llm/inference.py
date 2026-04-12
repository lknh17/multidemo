"""v03 文本生成推理: 贪心 / Top-k / Top-p，演示 KV Cache 加速"""
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import MiniGPT
from dataset import create_dataloaders
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50, strategy="top_p", temperature=0.8, top_k=50, top_p=0.9, use_cache=True, device="cpu"):
    model.eval()
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], device=device)
    kv_caches = None
    generated = list(ids)
    
    for _ in range(max_len):
        if use_cache and kv_caches is not None:
            logits, kv_caches = model(input_ids[:, -1:], kv_caches)
        else:
            logits, kv_caches = model(input_ids)
        
        logits = logits[:, -1, :] / temperature
        
        if strategy == "greedy":
            next_id = logits.argmax(-1)
        elif strategy == "top_k":
            top_vals, top_idx = logits.topk(top_k)
            probs = F.softmax(top_vals, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_id = top_idx.gather(-1, idx).squeeze(-1)
        elif strategy == "top_p":
            sorted_logits, sorted_idx = logits.sort(descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = probs.cumsum(-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            probs = F.softmax(sorted_logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_id = sorted_idx.gather(-1, idx).squeeze(-1)
        
        if next_id.item() == tokenizer.eos_id:
            break
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(generated)

def main():
    device = get_device()
    _, _, tokenizer = create_dataloaders(config)
    model = MiniGPT(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt):
        load_checkpoint(model, ckpt, device=str(device))
    
    prompts = ["the cat", "deep learning", "machine"]
    for strategy in ["greedy", "top_k", "top_p"]:
        print(f"\n{'='*40} {strategy} {'='*40}")
        for p in prompts:
            text = generate(model, tokenizer, p, strategy=strategy, device=device)
            print(f"  [{p}] → {text}")
    
    # KV Cache 速度对比
    print(f"\n{'='*40} KV Cache 速度对比 {'='*40}")
    import time
    for use_cache in [False, True]:
        start = time.time()
        for _ in range(10):
            generate(model, tokenizer, "the", max_len=50, use_cache=use_cache, device=device)
        elapsed = time.time() - start
        print(f"  use_cache={use_cache}: {elapsed:.3f}s (10 runs)")

if __name__ == "__main__":
    main()
