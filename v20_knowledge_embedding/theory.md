# V20 - 知识增强多模态嵌入：数学原理

## 1. 知识图谱嵌入

### 1.1 TransE

**核心思想**：将关系建模为实体嵌入空间中的平移操作。

对于三元组 $(h, r, t)$（头实体、关系、尾实体），TransE 期望：

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
$$

**评分函数**：

$$
f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_{p}
$$

其中 $p \in \{1, 2\}$ 为 $L_p$ 范数。

**训练目标**（Margin-based Ranking Loss）：

$$
\mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h',r,t') \in S'} \max(0, \gamma + f(h,r,t) - f(h',r,t'))
$$

- $S$：正样本三元组集合
- $S'$：负采样三元组集合（随机替换头或尾实体）
- $\gamma$：margin 超参数

**局限**：TransE 无法很好处理一对多（1-N）、多对一（N-1）、多对多（N-N）关系。

### 1.2 TransR

**改进思想**：不同关系在不同的语义空间中。为每个关系 $r$ 学习一个投影矩阵 $\mathbf{M}_r$。

$$
\mathbf{h}_r = \mathbf{M}_r \mathbf{h}, \quad \mathbf{t}_r = \mathbf{M}_r \mathbf{t}
$$

**评分函数**：

$$
f_r(h, r, t) = -\|\mathbf{h}_r + \mathbf{r} - \mathbf{t}_r\|_2^2
$$

其中 $\mathbf{M}_r \in \mathbb{R}^{d_r \times d_e}$ 将实体从 $d_e$ 维空间投影到 $d_r$ 维关系空间。

**优势**：每个关系有独立的语义空间，可以更好地建模复杂关系。

## 2. 实体链接（Entity Linking）

### 2.1 Mention 检测

给定文本序列 $\mathbf{x} = (x_1, \ldots, x_n)$，检测可能的实体提及 span：

$$
P(\text{mention} | x_i, x_j) = \sigma(\mathbf{W}_m [\mathbf{h}_i; \mathbf{h}_j; \mathbf{h}_i \odot \mathbf{h}_j])
$$

其中 $\mathbf{h}_i, \mathbf{h}_j$ 是上下文编码的 span 起止位置表示。

### 2.2 候选排序

对于每个检测到的 mention $m$，从候选实体集合 $\mathcal{C}_m$ 中选择最佳匹配：

$$
\text{score}(m, e) = \mathbf{h}_m^T \mathbf{W}_e \mathbf{e} + \mathbf{v}^T \phi(m, e)
$$

- $\mathbf{h}_m$：mention 的上下文表示
- $\mathbf{e}$：候选实体的 KG 嵌入
- $\phi(m, e)$：额外特征（字符串相似度、先验概率等）

**训练目标**（Cross-Entropy）：

$$
\mathcal{L}_{link} = -\sum_m \log \frac{\exp(\text{score}(m, e^*))}{\sum_{e \in \mathcal{C}_m} \exp(\text{score}(m, e))}
$$

## 3. KG 增强注意力

### 3.1 实体嵌入注入

将知识图谱的实体嵌入作为额外的 Key-Value 对注入到 Transformer 注意力中：

**标准注意力**：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

**KG 增强注意力**：

$$
K_{aug} = [K; K_{kg}], \quad V_{aug} = [V; V_{kg}]
$$

$$
\text{KGAttn}(Q, K, V, K_{kg}, V_{kg}) = \text{softmax}\left(\frac{Q [K; K_{kg}]^T}{\sqrt{d}}\right) [V; V_{kg}]
$$

其中 $K_{kg}, V_{kg}$ 由关联实体的 KG 嵌入经过线性变换得到：

$$
K_{kg} = \mathbf{W}_k^{kg} \mathbf{E}_{linked}, \quad V_{kg} = \mathbf{W}_v^{kg} \mathbf{E}_{linked}
$$

### 3.2 门控融合

为避免无关实体的噪声，使用门控机制控制 KG 信息的注入：

$$
\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{visual}; \mathbf{h}_{kg}])
$$

$$
\mathbf{h}_{fused} = (1 - \mathbf{g}) \odot \mathbf{h}_{visual} + \mathbf{g} \odot \mathbf{h}_{kg}
$$

## 4. GNN 消息传递

### 4.1 图注意力网络（GAT-style）

在知识图谱上执行消息传递，聚合邻居信息：

**消息计算**：

$$
\mathbf{m}_{j \to i} = \mathbf{W}_v \mathbf{h}_j + \mathbf{W}_r \mathbf{r}_{(j,i)}
$$

**注意力权重**：

$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_q \mathbf{h}_i; \mathbf{m}_{j \to i}]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_q \mathbf{h}_i; \mathbf{m}_{k \to i}]))}
$$

**节点更新**：

$$
\mathbf{h}_i^{(l+1)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{m}_{j \to i}\right)
$$

### 4.2 多跳推理

通过堆叠多层 GNN 实现多跳推理，$L$ 层 GNN 可以聚合 $L$ 跳邻居的信息：

$$
\mathbf{h}_i^{(L)} = \text{GNN}^{(L)}(\mathbf{h}_i^{(L-1)}, \{\mathbf{h}_j^{(L-1)} | j \in \mathcal{N}(i)\})
$$

## 5. 知识蒸馏

### 5.1 从 KG 到视觉模型的蒸馏

教师模型（KG-aware）的知识迁移到学生模型（纯视觉）：

**软标签蒸馏**：

$$
\mathcal{L}_{distill} = T^2 \cdot \text{KL}\left(\frac{\mathbf{z}_s}{T} \| \frac{\mathbf{z}_t}{T}\right)
$$

其中 $T$ 是温度参数，$\mathbf{z}_s, \mathbf{z}_t$ 分别是学生和教师的 logits。

**特征蒸馏**：

$$
\mathcal{L}_{feat} = \|\mathbf{f}_{student} - \text{sg}(\mathbf{W}_{proj} \mathbf{f}_{teacher})\|_2^2
$$

**总损失**：

$$
\mathcal{L} = (1 - \alpha) \mathcal{L}_{task} + \alpha \mathcal{L}_{distill} + \beta \mathcal{L}_{feat}
$$

### 5.2 知识增强的表示对齐

将 KG 嵌入空间与视觉嵌入空间对齐：

$$
\mathcal{L}_{align} = \sum_{(v, e) \in \mathcal{P}} \left(1 - \frac{\mathbf{f}_v \cdot \mathbf{e}}{\|\mathbf{f}_v\| \|\mathbf{e}\|}\right)
$$

其中 $\mathcal{P}$ 是图像-实体的正样本对。
