"""
V20 - 知识图谱核心模块
======================
1. TransE / TransR 知识图谱嵌入
2. SimpleGNN 图神经网络推理
3. EntityLinker 实体链接（mention 检测 + 候选排序）
4. KGAttentionLayer 知识增强注意力

参考：
- TransE: Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)
- TransR: Learning Entity and Relation Embeddings for Knowledge Graph Completion (AAAI 2015)
- ERNIE: Enhanced Language Representation with Informative Entities (ACL 2019)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from config import KnowledgeGraphConfig, EntityLinkConfig


# ============================================================
#  TransE 嵌入
# ============================================================

class TransEEmbedding(nn.Module):
    """
    TransE 知识图谱嵌入
    
    核心：关系 = 嵌入空间中的平移
    对于正确三元组 (h, r, t)：h + r ≈ t
    """

    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__()
        self.config = config
        self.entity_embed = nn.Embedding(config.num_entities, config.d_model)
        self.relation_embed = nn.Embedding(config.num_relations, config.d_model)
        self.margin = config.margin
        self.norm_p = config.norm_p

        # 初始化：均匀分布到单位球
        nn.init.uniform_(self.entity_embed.weight, -6.0 / math.sqrt(config.d_model),
                        6.0 / math.sqrt(config.d_model))
        nn.init.uniform_(self.relation_embed.weight, -6.0 / math.sqrt(config.d_model),
                        6.0 / math.sqrt(config.d_model))
        # 归一化关系嵌入
        with torch.no_grad():
            self.relation_embed.weight.data = F.normalize(
                self.relation_embed.weight.data, p=2, dim=-1
            )

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算三元组评分
        Args:
            h, r, t: [B] 实体/关系 ID
        Returns:
            scores: [B] 负距离（越大越好）
        """
        h_emb = F.normalize(self.entity_embed(h), p=2, dim=-1)
        r_emb = self.relation_embed(r)
        t_emb = F.normalize(self.entity_embed(t), p=2, dim=-1)
        return -torch.norm(h_emb + r_emb - t_emb, p=self.norm_p, dim=-1)

    def compute_loss(self, pos_h: torch.Tensor, pos_r: torch.Tensor, pos_t: torch.Tensor,
                     neg_h: torch.Tensor, neg_r: torch.Tensor, neg_t: torch.Tensor) -> torch.Tensor:
        """Margin ranking loss"""
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        loss = F.relu(self.margin + neg_score - pos_score)
        return loss.mean()

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """获取实体嵌入"""
        return self.entity_embed(entity_ids)


# ============================================================
#  TransR 嵌入
# ============================================================

class TransREmbedding(nn.Module):
    """
    TransR 知识图谱嵌入
    
    改进 TransE：每个关系有独立的投影空间
    h_r = M_r * h, t_r = M_r * t
    评分：-||h_r + r - t_r||
    """

    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__()
        self.config = config
        entity_dim = config.d_model
        relation_dim = config.relation_space_dim

        self.entity_embed = nn.Embedding(config.num_entities, entity_dim)
        self.relation_embed = nn.Embedding(config.num_relations, relation_dim)
        # 每个关系一个投影矩阵：relation_dim x entity_dim
        self.proj_matrices = nn.Embedding(
            config.num_relations, relation_dim * entity_dim
        )
        self.margin = config.margin
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        nn.init.uniform_(self.entity_embed.weight, -6.0 / math.sqrt(entity_dim),
                        6.0 / math.sqrt(entity_dim))
        nn.init.uniform_(self.relation_embed.weight, -6.0 / math.sqrt(relation_dim),
                        6.0 / math.sqrt(relation_dim))
        # 投影矩阵初始化为近似单位矩阵
        nn.init.eye_(self.proj_matrices.weight.view(-1, relation_dim, entity_dim)
                     .data.mean(dim=0))  # 仅示意

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_emb = self.entity_embed(h)   # [B, d_e]
        r_emb = self.relation_embed(r)  # [B, d_r]
        t_emb = self.entity_embed(t)   # [B, d_e]

        # 关系投影矩阵
        M = self.proj_matrices(r).view(-1, self.relation_dim, self.entity_dim)  # [B, d_r, d_e]

        # 投影到关系空间
        h_r = torch.bmm(M, h_emb.unsqueeze(-1)).squeeze(-1)  # [B, d_r]
        t_r = torch.bmm(M, t_emb.unsqueeze(-1)).squeeze(-1)  # [B, d_r]

        return -torch.norm(h_r + r_emb - t_r, p=2, dim=-1)

    def compute_loss(self, pos_h: torch.Tensor, pos_r: torch.Tensor, pos_t: torch.Tensor,
                     neg_h: torch.Tensor, neg_r: torch.Tensor, neg_t: torch.Tensor) -> torch.Tensor:
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        loss = F.relu(self.margin + neg_score - pos_score)
        return loss.mean()

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.entity_embed(entity_ids)


# ============================================================
#  SimpleGNN（图注意力网络）
# ============================================================

class SimpleGNNLayer(nn.Module):
    """GAT 风格的 GNN 层，用于 KG 上的消息传递"""

    def __init__(self, d_model: int, n_heads: int, num_relations: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rel_embed = nn.Embedding(num_relations, d_model)

        self.msg_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.attn_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model, 1),
        )

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [N, D] 节点特征
            edge_index: [2, E] 边索引 (src, dst)
            edge_type: [E] 边的关系类型
        Returns:
            updated_features: [N, D]
        """
        N = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # 消息计算：src 特征 + 关系嵌入
        src_feat = node_features[src]                  # [E, D]
        rel_feat = self.rel_embed(edge_type)           # [E, D]
        messages = self.msg_net(torch.cat([src_feat, rel_feat], dim=-1))  # [E, D]

        # 注意力权重
        dst_feat = node_features[dst]                  # [E, D]
        attn_input = torch.cat([dst_feat, messages], dim=-1)  # [E, 2D]
        attn_raw = self.attn_net(attn_input).squeeze(-1)      # [E]

        # Softmax over neighbors（简化：按 dst 分组）
        attn_max = torch.zeros(N, device=node_features.device)
        attn_max.scatter_reduce_(0, dst, attn_raw, reduce='amax', include_self=False)
        attn_exp = torch.exp(attn_raw - attn_max[dst])
        attn_sum = torch.zeros(N, device=node_features.device)
        attn_sum.scatter_add_(0, dst, attn_exp)
        attn_weights = attn_exp / (attn_sum[dst] + 1e-8)      # [E]

        # 加权聚合
        weighted_msg = attn_weights.unsqueeze(-1) * messages    # [E, D]
        aggr = torch.zeros(N, self.d_model, device=node_features.device)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        # 残差 + LayerNorm
        node_features = self.norm(node_features + aggr)
        node_features = self.norm2(node_features + self.ffn(node_features))

        return node_features


class SimpleGNN(nn.Module):
    """多层 GNN，用于知识图谱推理"""

    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleGNNLayer(config.d_model, config.n_heads, config.num_relations)
            for _ in range(config.n_gnn_layers)
        ])

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            node_features = layer(node_features, edge_index, edge_type)
        return node_features


# ============================================================
#  实体链接器
# ============================================================

class EntityLinker(nn.Module):
    """
    实体链接：将文本中的 mention 链接到 KG 实体
    
    两阶段：
    1. Mention 检测：识别文本中可能指向实体的 span
    2. 候选排序：从候选实体中选择最匹配的
    """

    def __init__(self, config: EntityLinkConfig):
        super().__init__()
        self.config = config

        # 文本编码器
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model, 8, config.d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Mention 检测（span start/end 分类）
        self.start_classifier = nn.Linear(config.d_model, 1)
        self.end_classifier = nn.Linear(config.d_model, 1)

        # Span 表示
        self.span_proj = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # 候选排序
        self.mention_proj = nn.Linear(config.d_model, config.d_model)
        self.entity_proj = nn.Linear(config.d_model, config.d_model)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """编码文本序列"""
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(pos)
        return self.encoder(x)

    def detect_mentions(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检测 mention span
        Returns:
            span_repr: [B, N_spans, D]
            start_logits: [B, L]
            end_logits: [B, L]
        """
        start_logits = self.start_classifier(hidden_states).squeeze(-1)  # [B, L]
        end_logits = self.end_classifier(hidden_states).squeeze(-1)      # [B, L]

        # 简化：取 top-k start/end 组合
        B, L, D = hidden_states.shape
        k = min(5, L)
        start_topk = start_logits.topk(k, dim=-1).indices  # [B, k]
        end_topk = end_logits.topk(k, dim=-1).indices      # [B, k]

        # 构建 span 表示（取 start-end 对）
        spans = []
        for b in range(B):
            batch_spans = []
            for i in range(k):
                si = start_topk[b, i]
                ei = end_topk[b, i]
                if ei >= si and (ei - si) < self.config.max_mention_len:
                    h_start = hidden_states[b, si]
                    h_end = hidden_states[b, ei]
                    span_feat = torch.cat([h_start, h_end, h_start * h_end])
                    batch_spans.append(self.span_proj(span_feat))
            if not batch_spans:
                batch_spans.append(torch.zeros(D, device=hidden_states.device))
            spans.append(torch.stack(batch_spans))

        # Pad to same length
        max_spans = max(s.shape[0] for s in spans)
        padded = torch.zeros(B, max_spans, D, device=hidden_states.device)
        for b, s in enumerate(spans):
            padded[b, :s.shape[0]] = s

        return padded, start_logits, end_logits

    def rank_candidates(self, mention_repr: torch.Tensor,
                       candidate_embeds: torch.Tensor) -> torch.Tensor:
        """
        对候选实体排序
        Args:
            mention_repr: [B, N_mentions, D]
            candidate_embeds: [B, N_mentions, N_cand, D]
        Returns:
            scores: [B, N_mentions, N_cand]
        """
        mention_proj = self.mention_proj(mention_repr)  # [B, N, D]
        cand_proj = self.entity_proj(candidate_embeds)  # [B, N, C, D]
        scores = torch.einsum('bmd,bmcd->bmc', mention_proj, cand_proj)
        return scores

    def forward(self, token_ids: torch.Tensor,
                candidate_embeds: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden = self.encode_text(token_ids)
        span_repr, start_logits, end_logits = self.detect_mentions(hidden)

        outputs = {
            'span_repr': span_repr,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'hidden_states': hidden,
        }

        if candidate_embeds is not None:
            link_scores = self.rank_candidates(span_repr, candidate_embeds)
            outputs['link_scores'] = link_scores

        return outputs


# ============================================================
#  KG 注意力层
# ============================================================

class KGAttentionLayer(nn.Module):
    """
    知识增强注意力层
    
    将 KG 实体嵌入注入为额外的 Key-Value 对
    K_aug = [K; K_kg], V_aug = [V; V_kg]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # KG 的 KV 投影
        self.kg_k_proj = nn.Linear(d_model, d_model)
        self.kg_v_proj = nn.Linear(d_model, d_model)

        # 门控：控制 KG 信息的注入强度
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                kg_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入序列
            kg_embeds: [B, E, D] KG 实体嵌入（可选）
        Returns:
            output: [B, L, D]
        """
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if kg_embeds is not None:
            E = kg_embeds.shape[1]
            k_kg = self.kg_k_proj(kg_embeds).reshape(B, E, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v_kg = self.kg_v_proj(kg_embeds).reshape(B, E, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # 拼接 KG 的 KV
            k_aug = torch.cat([k, k_kg], dim=2)   # [B, H, L+E, d]
            v_aug = torch.cat([v, v_kg], dim=2)   # [B, H, L+E, d]
        else:
            k_aug = k
            v_aug = v

        # 注意力计算
        attn = torch.matmul(q, k_aug.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_aug)  # [B, H, L, d]
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)

        # 门控融合（如果有 KG 信息）
        if kg_embeds is not None:
            # 全局 KG 特征
            kg_global = kg_embeds.mean(dim=1, keepdim=True).expand(-1, L, -1)
            gate_val = self.gate(torch.cat([out, kg_global], dim=-1))
            out = gate_val * self.out_proj(out) + (1 - gate_val) * x
        else:
            out = self.out_proj(out)

        return out
