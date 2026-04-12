# V25 - 端到端广告多模态管线：数学原理

## 1. 全管线公式化

### 1.1 端到端管线

整个广告投放流程可表示为多阶段函数复合：

$$
\text{Serve}(q) = \text{Filter} \circ \text{Rank} \circ \text{Rerank} \circ \text{Retrieve} \circ \text{Encode}(q)
$$

其中 $q$ 为用户请求（含上下文特征），各阶段分别承担：

| 阶段 | 输入 | 输出 | 目标 |
|------|------|------|------|
| Encode | 原始多模态 | 统一 embedding | 语义表示 |
| Retrieve | query embedding | Top-K 候选 | 高召回 |
| Rerank | K 候选 | 排序分数 | 高精度 |
| Filter | 排序结果 | 安全/质量过滤 | 合规保障 |
| Serve | 最终候选 | 投放决策 | 收益最大化 |

### 1.2 多模态统一编码

$$
\mathbf{e}_{ad} = \text{Fuse}(\mathbf{e}_{vis}, \mathbf{e}_{txt}, \mathbf{e}_{aud})
$$

视觉编码（V01-V05）：

$$
\mathbf{e}_{vis} = \text{ViT}(\mathbf{I}) = \text{CLS}(\text{TransformerEncoder}(\text{PatchEmbed}(\mathbf{I})))
$$

文本编码（V03）：

$$
\mathbf{e}_{txt} = \text{MeanPool}(\text{TransformerEncoder}(\text{TokenEmbed}(\mathbf{T})))
$$

音频编码（V17）：

$$
\mathbf{e}_{aud} = \text{MLP}(\text{MeanPool}(\mathbf{A}))
$$

融合方式（V04, V09）：

$$
\text{Fuse}_{attn}(\mathbf{E}) = \text{CrossAttention}(\mathbf{Q}_{fuse}, [\mathbf{e}_{vis}; \mathbf{e}_{txt}; \mathbf{e}_{aud}])
$$

## 2. 多阶段检索

### 2.1 召回阶段（Recall）

使用近似最近邻（ANN）快速召回：

$$
\mathcal{C}_{recall} = \text{TopK}_{k_1}(\text{sim}(\mathbf{e}_q, \mathbf{e}_{ad})), \quad k_1 \sim 100
$$

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^T \mathbf{b}}{||\mathbf{a}||_2 \cdot ||\mathbf{b}||_2}
$$

### 2.2 精排阶段（Precision）

使用交叉注意力重排器（V13）：

$$
s_{rerank}(q, d) = \text{MLP}(\text{CrossEncoder}([q; \text{SEP}; d]))
$$

$$
\mathcal{C}_{rerank} = \text{TopK}_{k_2}(s_{rerank}), \quad k_2 \sim 20
$$

## 3. 排序学习（Learning to Rank）

### 3.1 LambdaMART

$$
\mathcal{L}_{lambda} = \sum_{(i,j): y_i > y_j} |\Delta \text{NDCG}_{ij}| \cdot \log(1 + e^{-(s_i - s_j)})
$$

### 3.2 Listwise 损失

$$
\mathcal{L}_{list} = -\sum_{i} P_i^{true} \log P_i^{pred}, \quad P_i = \frac{e^{s_i / \tau}}{\sum_j e^{s_j / \tau}}
$$

### 3.3 NDCG 指标

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

## 4. CTR 预估（DeepFM 风格）

### 4.1 模型结构

$$
\hat{y}_{CTR} = \sigma(y_{FM} + y_{DNN})
$$

FM 部分（二阶特征交叉，V04）：

$$
y_{FM} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

DNN 部分：

$$
y_{DNN} = \text{MLP}([\mathbf{e}_{user}; \mathbf{e}_{ad}; \mathbf{e}_{context}])
$$

### 4.2 损失函数

$$
\mathcal{L}_{CTR} = -\frac{1}{N} \sum_{n=1}^{N} [y_n \log \hat{y}_n + (1 - y_n) \log(1 - \hat{y}_n)]
$$

## 5. 多目标优化

### 5.1 Pareto 前沿

给定 $M$ 个目标函数 $\{f_1, ..., f_M\}$，Pareto 最优解满足：

$$
\nexists \mathbf{x}' : \forall m, f_m(\mathbf{x}') \leq f_m(\mathbf{x}^*) \wedge \exists m, f_m(\mathbf{x}') < f_m(\mathbf{x}^*)
$$

### 5.2 加权聚合

$$
s_{final}(d) = \alpha_{ctr} \cdot s_{ctr}(d) + \alpha_{rel} \cdot s_{rel}(d) + \alpha_{div} \cdot s_{div}(d) + \alpha_{fresh} \cdot s_{fresh}(d)
$$

多样性得分（MMR，V11）：

$$
s_{div}(d) = \lambda \cdot \text{rel}(d) - (1-\lambda) \max_{d' \in S} \text{sim}(d, d')
$$

新鲜度得分：

$$
s_{fresh}(d) = \exp(-\gamma \cdot \text{age}(d))
$$

### 5.3 多目标梯度下降（MGDA）

$$
\min_{\alpha} \left\| \sum_{m=1}^{M} \alpha_m \nabla_\theta \mathcal{L}_m \right\|^2, \quad \text{s.t. } \alpha_m \geq 0, \sum \alpha_m = 1
$$

## 6. 在线学习

### 6.1 增量更新

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\mathbf{x}_t, y_t; \theta_t)
$$

### 6.2 指数移动平均（EMA）

$$
\bar{\theta}_{t+1} = \beta \bar{\theta}_t + (1 - \beta) \theta_{t+1}
$$

### 6.3 遗忘因子

$$
\mathcal{L}_{online} = \sum_{t'=1}^{t} \gamma^{t-t'} \ell(\mathbf{x}_{t'}, y_{t'}; \theta)
$$

越近期的样本权重越大，实现模型对分布漂移的自适应。

## 7. 安全过滤（V24）

级联过滤策略：

$$
\text{Pass}(d) = \bigwedge_{k=1}^{K} [\text{SafetyCheck}_k(d) > \tau_k]
$$

多级检查：关键词 → 分类器 → 人工审核，兼顾效率与准确性。
