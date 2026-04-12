"""v11 Embedding 模型 + 提取器 + FAISS 检索"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from config import EmbeddingConfig

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from fusion import get_fusion_module
from model import MultiLayerEncoder


class EmbeddingModel(nn.Module):
    """端到端 Embedding 模型 (整合 DeepStack + 投影)"""
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        np_img = (cfg.image_size // cfg.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.img_cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.img_pos = nn.Parameter(torch.randn(1, np_img+1, cfg.d_model) * 0.02)
        self.img_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.txt_pos = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        self.txt_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        num_extract = len(cfg.layer_indices)
        self.fusion_img = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        self.fusion_txt = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/cfg.temperature)))
    
    def encode_image(self, images):
        B = images.size(0)
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat([self.img_cls.expand(B, -1, -1), x], dim=1)
        x = x + self.img_pos[:, :x.size(1)]
        _, inters = self.img_encoder.forward_with_intermediates(x, set(self.cfg.layer_indices))
        feats = [inters[l] for l in self.cfg.layer_indices]
        return F.normalize(self.fusion_img(feats), dim=-1)
    
    def encode_text(self, input_ids):
        x = self.tok_embed(input_ids) + self.txt_pos[:, :input_ids.size(1)]
        _, inters = self.txt_encoder.forward_with_intermediates(x, set(self.cfg.layer_indices))
        feats = [inters[l].mean(1) for l in self.cfg.layer_indices]
        return F.normalize(self.fusion_txt(feats), dim=-1)
    
    def forward(self, images, input_ids):
        return self.encode_image(images), self.encode_text(input_ids), self.logit_scale.exp()


class FAISSRetriever:
    """FAISS 向量检索封装"""
    def __init__(self, embed_dim, index_type="flat"):
        try:
            import faiss
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(embed_dim)  # 内积 (对归一化向量 = 余弦相似度)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(embed_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embed_dim, min(100, embed_dim), faiss.METRIC_INNER_PRODUCT)
            self.faiss = faiss
        except ImportError:
            print("FAISS not installed, using numpy brute force")
            self.index = None
            self.embeddings = None
    
    def build(self, embeddings: np.ndarray):
        if self.index is not None:
            if hasattr(self.index, 'train'):
                self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            self.embeddings = embeddings
    
    def search(self, queries: np.ndarray, top_k: int = 10):
        if self.index is not None:
            scores, indices = self.index.search(queries, top_k)
            return scores, indices
        else:
            sims = queries @ self.embeddings.T
            indices = np.argsort(-sims, axis=1)[:, :top_k]
            scores = np.take_along_axis(sims, indices, axis=1)
            return scores, indices


def compute_recall_at_k(retrieved_indices, gt_labels, query_labels, k):
    hits = 0
    for i in range(len(query_labels)):
        retrieved = gt_labels[retrieved_indices[i, :k]]
        if query_labels[i] in retrieved:
            hits += 1
    return hits / len(query_labels)

def compute_ndcg_at_k(retrieved_indices, gt_labels, query_labels, k):
    ndcg = 0
    for i in range(len(query_labels)):
        dcg = sum(1/np.log2(j+2) for j in range(k) if gt_labels[retrieved_indices[i, j]] == query_labels[i])
        idcg = sum(1/np.log2(j+2) for j in range(min(k, sum(gt_labels == query_labels[i]))))
        ndcg += dcg / max(idcg, 1e-8)
    return ndcg / len(query_labels)

def compute_mrr(retrieved_indices, gt_labels, query_labels):
    mrr = 0
    for i in range(len(query_labels)):
        for j in range(retrieved_indices.shape[1]):
            if gt_labels[retrieved_indices[i, j]] == query_labels[i]:
                mrr += 1 / (j + 1)
                break
    return mrr / len(query_labels)
