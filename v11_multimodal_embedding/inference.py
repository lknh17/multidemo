"""v11 端到端检索 Pipeline: 编码→索引→检索→评估"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from config import config
from model import EmbeddingModel, FAISSRetriever, compute_recall_at_k, compute_ndcg_at_k, compute_mrr
from dataset import create_dataloaders
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def encode_all(model, loader, device, mode="image"):
    model.eval()
    all_emb, all_labels = [], []
    for batch in loader:
        if mode == "image":
            emb = model.encode_image(batch["image"].to(device))
        else:
            emb = model.encode_text(batch["input_ids"].to(device))
        all_emb.append(emb.cpu()); all_labels.append(batch["label"])
    return torch.cat(all_emb).numpy(), torch.cat(all_labels).numpy()

def main():
    device = get_device()
    model = EmbeddingModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt): load_checkpoint(model, ckpt, device=str(device))
    
    _, gallery_loader, query_loader = create_dataloaders(config)
    
    print("=== 编码 Gallery ===")
    gallery_emb, gallery_labels = encode_all(model, gallery_loader, device, "image")
    print(f"Gallery: {gallery_emb.shape}")
    
    print("=== 编码 Queries ===")
    query_emb, query_labels = encode_all(model, query_loader, device, "text")
    print(f"Queries: {query_emb.shape}")
    
    print("=== 构建索引 & 检索 ===")
    retriever = FAISSRetriever(config.embed_dim)
    retriever.build(gallery_emb)
    scores, indices = retriever.search(query_emb, config.top_k)
    
    print("=== 评估结果 ===")
    for k in [1, 5, 10]:
        r = compute_recall_at_k(indices, gallery_labels, query_labels, k)
        print(f"  Recall@{k}: {r:.4f}")
    ndcg = compute_ndcg_at_k(indices, gallery_labels, query_labels, 10)
    mrr = compute_mrr(indices, gallery_labels, query_labels)
    print(f"  NDCG@10: {ndcg:.4f}")
    print(f"  MRR: {mrr:.4f}")

if __name__ == "__main__":
    main()
