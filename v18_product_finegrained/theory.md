# V18 - 商品理解与细粒度视觉：数学原理

## 1. 细粒度视觉识别

### 1.1 问题定义

细粒度识别 = 在同一大类内区分子类（如区分 200 种鸟、100 种汽车）。

挑战：类间差异小、类内差异大。关键在于捕获**判别性局部特征**。

### 1.2 双线性池化（Bilinear Pooling）

传统方法捕获局部特征交互：

$$
\mathbf{y} = \text{vec}(\mathbf{f}_A^T \mathbf{f}_B) \in \mathbb{R}^{D_A \times D_B}
$$

**紧凑双线性**（Compact Bilinear）：用随机投影降维：

$$
\mathbf{y}_{compact} = \text{FFT}^{-1}(\text{FFT}(\mathbf{p}_A) \odot \text{FFT}(\mathbf{p}_B))
$$

### 1.3 注意力裁剪（Attention Cropping）

基于注意力权重定位判别性区域：

$$
\text{bbox} = \text{BoundingBox}(\{(i,j) : A_{ij} > \theta\})
$$

$$
\mathbf{f}_{local} = \text{Encoder}(\text{Crop}(\mathbf{I}, \text{bbox}))
$$

## 2. 多粒度特征学习

### 2.1 全局-局部联合

$$
\mathbf{f} = [\mathbf{f}_{global}; \mathbf{f}_{part_1}; ...; \mathbf{f}_{part_K}]
$$

**全局特征**：整图 CLS token
**局部特征**：Top-K 注意力区域

### 2.2 多粒度损失

$$
\mathcal{L} = \mathcal{L}_{cls}^{global} + \sum_{k=1}^{K} \mathcal{L}_{cls}^{part_k} + \lambda \mathcal{L}_{diversity}
$$

多样性正则化：确保不同 part 关注不同区域：

$$
\mathcal{L}_{diversity} = ||A^T A - I||_F^2
$$

## 3. 商品属性提取

### 3.1 多标签分类

商品同时具有多个属性，使用独立的 Sigmoid：

$$
P(y_k = 1 | \mathbf{x}) = \sigma(\mathbf{w}_k^T \mathbf{f}(\mathbf{x}) + b_k)
$$

$$
\mathcal{L}_{attr} = -\sum_{k} [y_k \log p_k + (1 - y_k) \log(1 - p_k)]
$$

### 3.2 层次化分类

类目通常有层次结构（大类→中类→小类）：

$$
P(y_{fine} | \mathbf{x}) = P(y_{fine} | y_{coarse}, \mathbf{x}) \cdot P(y_{coarse} | \mathbf{x})
$$

### 3.3 属性间相关性建模

使用 GNN 或 Transformer 建模属性之间的依赖：

$$
\mathbf{H}_{attr} = \text{TransformerDecoder}(\mathbf{Q}_{attr}, \mathbf{H}_{visual})
$$

## 4. 商品图像质量评估

### 4.1 无参考质量评估（No-Reference IQA）

$$
q = f_{quality}(\text{Encoder}(\mathbf{I})) \in [0, 1]
$$

### 4.2 多维度评分

$$
\mathbf{q} = [q_{清晰度}, q_{曝光}, q_{构图}, q_{美感}, q_{合规}]
$$

### 4.3 排序学习

$$
\mathcal{L}_{rank} = \max(0, \epsilon - (q_i - q_j)) \quad \text{if } \mathbf{I}_i \succ \mathbf{I}_j
$$

## 5. 跨域商品匹配

### 5.1 度量学习

$$
d(\mathbf{x}_i, \mathbf{x}_j) = ||\mathbf{f}(\mathbf{x}_i) - \mathbf{f}(\mathbf{x}_j)||_2
$$

### 5.2 ArcFace Loss

$$
\mathcal{L} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
$$

其中 $m$ 是角度 margin，$s$ 是缩放因子。强制类内更紧凑、类间更分散。
