# V19 - 层级标签理解：数学原理

## 1. 分类学树（Taxonomy Tree）

### 1.1 树结构定义

分类学树 $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ 是一棵有根树：
- $\mathcal{V}$：节点集合（所有标签）
- $\mathcal{E}$：边集合（父子关系）
- 根节点 $r$ 是最粗粒度的分类

层级结构：$L_0$（行业）→ $L_1$（品类）→ $L_2$（子品类）→ $L_3$（SKU）

**父子约束**：如果一个样本属于细粒度标签 $c$，则必然属于其所有祖先节点：

$$
y_c = 1 \implies y_{\text{parent}(c)} = 1 \implies \ldots \implies y_r = 1
$$

### 1.2 路径概率分解

从根到叶的分类概率可以分解为条件概率链：

$$
P(y = c | \mathbf{x}) = \prod_{k=0}^{D-1} P(v_{k+1} | v_k, \mathbf{x})
$$

其中 $v_0 = r, v_1, \ldots, v_D = c$ 是从根到叶的路径。

## 2. 层级 Softmax（Hierarchical Softmax）

### 2.1 基本思想

传统 Softmax 在类别数 $C$ 很大时计算量为 $O(C)$。层级 Softmax 利用树结构将计算复杂度降至 $O(D \cdot \max_k |\text{children}(v_k)|)$。

**逐层条件 Softmax**：

$$
P(v_{k+1} | v_k, \mathbf{x}) = \frac{\exp(\mathbf{w}_{v_{k+1}}^T \mathbf{h}_k / \tau)}{\sum_{c \in \text{children}(v_k)} \exp(\mathbf{w}_c^T \mathbf{h}_k / \tau)}
$$

其中：
- $\mathbf{h}_k$ 是第 $k$ 层的特征表示
- $\mathbf{w}_c$ 是类别 $c$ 的权重向量
- $\tau$ 是温度参数

### 2.2 损失函数

**层级交叉熵损失**：

$$
\mathcal{L}_{hier} = -\sum_{k=0}^{D-1} \lambda_k \log P(v_{k+1}^* | v_k^*, \mathbf{x})
$$

其中 $\lambda_k$ 是每层的权重，通常细粒度层权重更大：$\lambda_k = \frac{k+1}{\sum_{i=1}^{D} i}$

## 3. 标签传播（Label Propagation）

### 3.1 图上的标签传播

将标签体系构建为图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$：
- 邻接矩阵 $\mathbf{A} \in \{0, 1\}^{N \times N}$
- 度矩阵 $\mathbf{D} = \text{diag}(\sum_j A_{ij})$

**归一化传播**：

$$
\mathbf{Y}^{(t+1)} = \alpha \hat{\mathbf{A}} \mathbf{Y}^{(t)} + (1 - \alpha) \mathbf{Y}^{(0)}
$$

其中 $\hat{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ 是对称归一化邻接矩阵，$\alpha \in [0, 1)$ 是传播衰减率。

### 3.2 GNN 标签传播

使用图神经网络学习传播函数：

$$
\mathbf{H}^{(l+1)} = \sigma\left(\hat{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)
$$

其中 $\mathbf{W}^{(l)}$ 是可学习参数。

**消息传递机制**：

$$
\mathbf{h}_v^{(l+1)} = \phi\left(\mathbf{h}_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} \psi(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)})\right)
$$

## 4. 双曲空间标签嵌入

### 4.1 Poincaré 球模型

双曲空间天然适合表示层级结构。Poincaré 球模型定义在开球 $\mathbb{B}^d = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| < 1\}$ 上。

**双曲距离**：

$$
d_{\mathcal{P}}(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left(1 + 2 \frac{\|\mathbf{u} - \mathbf{v}\|^2}{(1 - \|\mathbf{u}\|^2)(1 - \|\mathbf{v}\|^2)}\right)
$$

**关键性质**：
- 靠近原点的点对应粗粒度概念（根节点）
- 靠近边界的点对应细粒度概念（叶节点）
- 层级关系自然编码为径向距离

### 4.2 Möbius 运算

**Möbius 加法**（双曲空间中的向量加法）：

$$
\mathbf{u} \oplus_c \mathbf{v} = \frac{(1 + 2c\langle\mathbf{u},\mathbf{v}\rangle + c\|\mathbf{v}\|^2)\mathbf{u} + (1 - c\|\mathbf{u}\|^2)\mathbf{v}}{1 + 2c\langle\mathbf{u},\mathbf{v}\rangle + c^2\|\mathbf{u}\|^2\|\mathbf{v}\|^2}
$$

**指数映射**（欧氏空间→双曲空间）：

$$
\exp_\mathbf{x}^c(\mathbf{v}) = \mathbf{x} \oplus_c \left(\tanh\left(\sqrt{c}\frac{\lambda_\mathbf{x}^c \|\mathbf{v}\|}{2}\right) \frac{\mathbf{v}}{\sqrt{c}\|\mathbf{v}\|}\right)
$$

其中 $\lambda_\mathbf{x}^c = \frac{2}{1 - c\|\mathbf{x}\|^2}$ 是共形因子。

### 4.3 层级嵌入损失

**层级对比损失**：

$$
\mathcal{L}_{hyp} = \sum_{(i,j) \in \mathcal{P}} \max(0, d_{\mathcal{P}}(\mathbf{u}_i, \mathbf{v}_j) - d_{\mathcal{P}}(\mathbf{u}_i, \mathbf{v}_{j'}) + m)
$$

其中 $\mathcal{P}$ 是正样本对集合（父子/祖孙关系），$j'$ 是负样本。

## 5. 多标签分类与层级约束

### 5.1 层级约束 BCE

基础多标签损失为 Binary Cross-Entropy：

$$
\mathcal{L}_{BCE} = -\frac{1}{C} \sum_{c=1}^{C} \left[y_c \log \hat{y}_c + (1 - y_c) \log(1 - \hat{y}_c)\right]
$$

**层级一致性约束**：预测子节点为正时，其祖先也应为正：

$$
\mathcal{L}_{consist} = \sum_{c} \sum_{a \in \text{ancestors}(c)} \max(0, \hat{y}_c - \hat{y}_a)
$$

### 5.2 标签平滑

对层级标签进行平滑，将部分概率质量分配给相近标签：

$$
\tilde{y}_c = (1 - \epsilon) y_c + \epsilon \cdot \frac{\sum_{c' \in \text{siblings}(c)} y_{c'}}{|\text{siblings}(c)|}
$$

### 5.3 分类学损失（Taxonomic Loss）

根据标签间的树距离加权损失：

$$
\mathcal{L}_{tax} = \sum_{c} w_{c,\hat{c}} \cdot \ell(y_c, \hat{y}_c)
$$

其中树距离权重：

$$
w_{c, c'} = \frac{d_{tree}(c, c')}{D_{max}}
$$

$d_{tree}(c, c')$ 是标签 $c$ 和 $c'$ 在分类树上的最短路径长度。
