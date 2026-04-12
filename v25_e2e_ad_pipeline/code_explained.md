# V25 - 端到端广告多模态管线：代码详解

## 1. 管线核心模块（pipeline_modules.py）

### 1.1 AdEncoder — 多模态统一编码器

```python
class AdEncoder(nn.Module):
    def forward(self, images, text_ids, audio_feats=None):
        # 1. 视觉编码（V01-V05 ViT）
        vis_feat = self.vis_encoder(images)       # [B, D]

        # 2. 文本编码（V03 Transformer）
        txt_feat = self.txt_encoder(text_ids)     # [B, D]

        # 3. 音频编码（V17 MLP）
        aud_feat = self.aud_encoder(audio_feats)  # [B, D]

        # 4. 注意力融合（V04 Cross-Attention）
        modalities = stack([vis, txt, aud])       # [B, 3, D]
        fused = self.fusion_attn(query, modalities)
        return F.normalize(fused, dim=-1)
```

关键：三路编码器独立提取特征，再用注意力融合为统一 embedding。

### 1.2 AdMatcher — 检索匹配器

```python
class AdMatcher(nn.Module):
    def retrieve(self, query_emb, index_embs, top_k=100):
        # ANN 近似检索：余弦相似度
        scores = query_emb @ index_embs.T
        top_indices = scores.topk(top_k).indices
        return top_indices, scores[top_indices]

    def rerank(self, query_emb, candidate_embs):
        # 交叉注意力精排（V13）
        cross_scores = self.cross_encoder(query_emb, candidate_embs)
        return cross_scores.sort(descending=True)
```

### 1.3 SafetyFilter — 安全级联过滤

```python
class SafetyFilter(nn.Module):
    def forward(self, embeddings):
        # 级联检查（V24）
        # 1. 快速关键词过滤
        keyword_safe = self.keyword_check(embeddings)
        # 2. 分类器过滤
        classifier_safe = self.safety_classifier(embeddings)
        # 3. 综合判定
        return keyword_safe & (classifier_safe > threshold)
```

### 1.4 QualityGate — 质量门控

```python
class QualityGate(nn.Module):
    def forward(self, embeddings):
        # 多维度质量评分（V18 风格）
        scores = self.quality_head(embeddings)  # [B, 5]
        overall = self.overall_head(cat([emb, scores]))
        return overall > self.threshold
```

## 2. 主模型（model.py）

### 2.1 E2EAdPipeline — 全管线

```python
class E2EAdPipeline(nn.Module):
    def forward(self, query_images, query_texts, ad_images, ad_texts, ...):
        # 阶段 1: Encode
        query_emb = self.encoder(query_images, query_texts)
        ad_embs = self.encoder(ad_images, ad_texts)

        # 阶段 2: Retrieve (ANN Top-K)
        retrieved_idx, recall_scores = self.matcher.retrieve(query_emb, ad_embs)

        # 阶段 3: Rerank (Cross-Attention)
        rerank_scores = self.matcher.rerank(query_emb, ad_embs[retrieved_idx])

        # 阶段 4: Safety Filter
        safe_mask = self.safety_filter(ad_embs[top_indices])

        # 阶段 5: Quality Gate
        quality_mask = self.quality_gate(ad_embs[top_indices])

        # 阶段 6: Serve
        final_mask = safe_mask & quality_mask
        return rerank_scores[final_mask]
```

### 2.2 CTRPredictor — DeepFM 风格

```python
class CTRPredictor(nn.Module):
    def forward(self, user_emb, ad_emb, context_emb):
        # FM: 二阶特征交叉
        combined = cat([user_emb, ad_emb, context_emb])
        fm_out = self.fm_layer(combined)

        # DNN: 深层非线性
        dnn_out = self.dnn(combined)

        # 融合
        ctr = sigmoid(fm_out + dnn_out)
        return ctr
```

### 2.3 MultiObjectiveRanker — 多目标排序

```python
class MultiObjectiveRanker(nn.Module):
    def forward(self, ad_emb, user_emb, context_emb):
        features = cat([ad_emb, user_emb, context_emb])

        ctr_score = self.ctr_head(features)
        rel_score = self.relevance_head(features)
        div_score = self.diversity_head(features)
        fresh_score = self.freshness_head(features)

        # 加权聚合
        final = (w_ctr * ctr + w_rel * rel
                 + w_div * div + w_fresh * fresh)
        return final
```

### 2.4 OnlineLearner — 增量更新

```python
class OnlineLearner:
    def update(self, new_sample):
        # 单样本梯度更新
        loss = self.model.compute_loss(new_sample)
        loss.backward()
        self.optimizer.step()

        # EMA 平滑参数
        for ema_p, p in zip(self.ema_model.params, self.model.params):
            ema_p.data = beta * ema_p.data + (1 - beta) * p.data
```

## 3. 训练脚本（train.py）

三种训练模式：

| 模式 | 训练目标 | 损失函数 |
|------|----------|----------|
| pipeline | 端到端对比学习 | InfoNCE + 安全分类 |
| ctr | CTR 预估 | Binary Cross-Entropy |
| ranker | 多目标排序 | Listwise + 多目标加权 |

## 4. 推理实验（inference.py）

| 实验 | 衡量指标 |
|------|----------|
| 延迟分解 | 各阶段耗时占比 |
| Recall@K | 不同 K 下的召回率 |
| Pareto 分析 | 多目标权重下的 Pareto 前沿 |
| 在线学习 | 累积 CTR 收敛曲线 |

## 5. 关键设计决策

1. **统一 embedding 空间**：所有模态映射到同一向量空间，便于 ANN 检索
2. **级联架构**：粗筛→精排→过滤，兼顾效率与效果
3. **多目标平衡**：加权聚合 + Pareto 分析，避免单一目标主导
4. **在线学习**：EMA + 遗忘因子适应数据分布漂移
5. **安全前置**：安全过滤作为硬约束，不可被排序分数覆盖
