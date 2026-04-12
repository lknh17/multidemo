# V20 - 知识增强多模态嵌入：代码详解

## 1. KG 嵌入模块（kg_modules.py）

### 1.1 TransE 嵌入

```python
class TransEEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        self.entity_embed = nn.Embedding(num_entities, dim)
        self.relation_embed = nn.Embedding(num_relations, dim)
        self.margin = margin

    def score(self, h, r, t):
        # TransE 评分：h + r ≈ t → 距离越小越好
        h_emb = self.entity_embed(h)
        r_emb = self.relation_embed(r)
        t_emb = self.entity_embed(t)
        return -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

    def compute_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # Margin ranking loss
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        loss = F.relu(self.margin - pos_score + neg_score)
        return loss.mean()
```

**关键理解**：TransE 将关系视为嵌入空间中的平移向量。正确三元组 $(h,r,t)$ 满足 $h+r\approx t$，错误三元组的距离应该更大。margin loss 确保正负样本之间有足够间隔。

### 1.2 TransR 嵌入

```python
class TransREmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim):
        self.entity_embed = nn.Embedding(num_entities, entity_dim)
        self.relation_embed = nn.Embedding(num_relations, relation_dim)
        # 每个关系一个投影矩阵
        self.proj_matrices = nn.Embedding(num_relations, relation_dim * entity_dim)

    def score(self, h, r, t):
        h_emb = self.entity_embed(h)
        r_emb = self.relation_embed(r)
        t_emb = self.entity_embed(t)
        # 获取关系特定的投影矩阵
        M = self.proj_matrices(r).view(-1, self.relation_dim, self.entity_dim)
        # 投影到关系空间
        h_r = torch.bmm(M, h_emb.unsqueeze(-1)).squeeze(-1)
        t_r = torch.bmm(M, t_emb.unsqueeze(-1)).squeeze(-1)
        return -torch.norm(h_r + r_emb - t_r, p=2, dim=-1)
```

**为什么需要 TransR**：不同关系应该在不同的语义空间中比较。例如"出生地"和"工作于"关注实体的不同语义侧面，用单一空间难以区分。

### 1.3 SimpleGNN（图神经网络）

```python
class SimpleGNN(nn.Module):
    """GAT 风格的 GNN，用于知识图谱推理"""

    def forward(self, node_features, edge_index, edge_type):
        for layer in self.layers:
            # 消息计算
            src_features = node_features[edge_index[0]]
            rel_features = self.rel_embed(edge_type)
            messages = self.msg_net(src_features + rel_features)

            # 注意力加权
            dst_features = node_features[edge_index[1]]
            attn = self.attn_net(torch.cat([dst_features, messages], -1))
            attn = softmax(attn, edge_index[1])

            # 聚合
            aggr = scatter_sum(attn * messages, edge_index[1])
            node_features = node_features + aggr
```

**理解消息传递**：每个节点从邻居收集信息，关系类型影响消息内容，注意力机制决定每个邻居的重要性。

## 2. 实体链接（kg_modules.py）

### 2.1 Mention 检测

```python
class EntityLinker(nn.Module):
    def detect_mentions(self, hidden_states):
        # 对每个可能的 span 计算是否为 mention
        B, L, D = hidden_states.shape
        # 枚举所有可能的 span (start, end)
        start_logits = self.start_classifier(hidden_states)  # [B, L, 1]
        end_logits = self.end_classifier(hidden_states)      # [B, L, 1]

        # Span 表示 = [start; end; start⊙end]
        spans = self.compute_span_representations(
            hidden_states, start_logits, end_logits
        )
        return spans, start_logits, end_logits
```

### 2.2 候选排序

```python
def rank_candidates(self, mention_repr, candidate_entities):
    """
    mention_repr: [B, N_mentions, D] mention 上下文表示
    candidate_entities: [B, N_mentions, N_cand, D] 候选实体的 KG 嵌入
    """
    # 双线性打分
    mention_proj = self.mention_proj(mention_repr)  # [B, N, D]
    scores = torch.einsum(
        'bmd,bmcd->bmc',
        mention_proj, candidate_entities
    )  # [B, N_mentions, N_candidates]
    return scores
```

## 3. KG 增强视觉模型（model.py）

### 3.1 KGAttentionLayer

```python
class KGAttentionLayer(nn.Module):
    """将 KG 实体嵌入注入为额外 KV 对"""

    def forward(self, x, kg_keys, kg_values):
        # 标准 QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拼接 KG 的 KV
        k_aug = torch.cat([k, kg_keys], dim=1)    # [B, L+E, D]
        v_aug = torch.cat([v, kg_values], dim=1)   # [B, L+E, D]

        # 注意力计算
        attn = (q @ k_aug.transpose(-2, -1)) / sqrt(d)
        attn = softmax(attn, dim=-1)
        output = attn @ v_aug
        return output
```

**核心创新**：不修改 Query，而是扩展 Key-Value 空间，让模型可以选择性地关注 KG 中的实体信息。

### 3.2 KGEnhancedVisualModel

```python
class KGEnhancedVisualModel(nn.Module):
    """ViT + KG 注意力"""

    def forward(self, images, entity_ids):
        # 1. ViT 提取视觉特征
        patches = self.patch_embed(images)  # [B, N, D]

        # 2. 获取关联实体的 KG 嵌入
        entity_embs = self.kg_embed(entity_ids)  # [B, E, D]
        kg_keys = self.kg_k_proj(entity_embs)
        kg_values = self.kg_v_proj(entity_embs)

        # 3. KG 增强 Transformer
        x = patches
        for block in self.blocks:
            x = block(x, kg_keys, kg_values)

        return x
```

### 3.3 门控融合

```python
def gated_fusion(self, visual_feat, kg_feat):
    """门控融合视觉和 KG 特征"""
    gate = torch.sigmoid(
        self.gate_proj(torch.cat([visual_feat, kg_feat], dim=-1))
    )
    fused = (1 - gate) * visual_feat + gate * kg_feat
    return fused
```

## 4. 知识蒸馏（model.py）

### 4.1 KnowledgeDistillModel

```python
class KnowledgeDistillModel(nn.Module):
    """从 KG-aware 教师蒸馏到纯视觉学生"""

    def compute_distill_loss(self, student_logits, teacher_logits, T=4.0):
        # 软标签 KL 散度
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        return T * T * kl_loss

    def compute_feature_loss(self, student_feat, teacher_feat):
        # 特征对齐
        projected = self.align_proj(teacher_feat)
        return F.mse_loss(student_feat, projected.detach())
```

**温度 T 的作用**：T 越大，softmax 输出越平滑，传递更多 dark knowledge（类别间的相对关系）。T=4 是常用选择。

## 5. 训练流程

### 5.1 KG 嵌入训练

```python
# 正样本：真实三元组 (h, r, t)
# 负采样：随机替换头或尾实体
for h, r, t in kg_dataloader:
    # 负采样
    neg_h, neg_t = random_corrupt(h, r, t, num_entities)
    loss = model.compute_loss(h, r, t, neg_h, r, neg_t)
```

### 5.2 KG 增强视觉训练

```python
for images, entity_ids, labels in dataloader:
    # 视觉特征 + KG 嵌入 → 融合特征
    outputs = kg_visual_model(images, entity_ids)
    # 分类 / 检索损失
    loss = criterion(outputs['logits'], labels)
```

### 5.3 知识蒸馏

```python
# 教师：KG-enhanced 模型（冻结）
# 学生：纯视觉模型
teacher_out = teacher(images, entity_ids)
student_out = student(images)

loss = (1 - alpha) * task_loss + alpha * distill_loss + beta * feat_loss
```
