# V16 - OCR 与文档理解：数学原理

## 1. OCR 文字检测

### 1.1 DBNet（Differentiable Binarization）

传统流程：分割图 → 固定阈值二值化 → 后处理。**DBNet 核心创新**：学习自适应阈值。

**Probability Map**：分割网络输出概率图 $P \in [0, 1]^{H \times W}$

**Threshold Map**：同时预测逐像素阈值 $T \in [0, 1]^{H \times W}$

**可微二值化（Differentiable Binarization, DB）**：

$$
\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}
$$

其中 $k$ 为放大因子（通常 $k=50$）。当 $k \to \infty$ 时等价于硬二值化，但可微分可训练。

**损失函数**：

$$
\mathcal{L} = \mathcal{L}_{prob} + \alpha \mathcal{L}_{thresh} + \beta \mathcal{L}_{binary}
$$

- $\mathcal{L}_{prob}$：BCE Loss for probability map
- $\mathcal{L}_{thresh}$：L1 Loss for threshold map
- $\mathcal{L}_{binary}$：BCE Loss for binary map

### 1.2 EAST（Efficient and Accurate Scene Text）

直接回归文字框的几何参数：

每个像素预测：
- 置信度 $p_i \in [0, 1]$
- 到文字框四边的距离 $(d_{top}, d_{right}, d_{bottom}, d_{left})$
- 旋转角度 $\theta_i$

优势：无需 Anchor，单阶段检测。

## 2. 文字识别（Scene Text Recognition）

### 2.1 CRNN（CNN + RNN + CTC）

**架构**：

$$
\text{Image} \xrightarrow{\text{CNN}} \text{Feature Map} \xrightarrow{\text{Reshape}} \text{Sequence} \xrightarrow{\text{BiLSTM}} \text{Prediction}
$$

**CTC（Connectionist Temporal Classification）Loss**：

将识别视为序列标注问题。定义映射 $\mathcal{B}: \text{(含重复和空白的路径)} \to \text{标签序列}$

$$
P(\mathbf{y} | \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} p(\pi_t | \mathbf{x})
$$

CTC Loss = $-\log P(\mathbf{y}^* | \mathbf{x})$

**局限**：条件独立假设，不建模字符间依赖。

### 2.2 ABINet（Autonomous, Bidirectional, Iterative）

**三个组件**：
1. **Vision Model (V)**：CNN + Transformer 提取视觉特征
2. **Language Model (L)**：双向语言模型建模字符间关系
3. **Fusion Model (F)**：融合视觉和语言信息

**迭代优化**：

$$
\mathbf{y}^{(0)} = V(\mathbf{x}), \quad \mathbf{y}^{(t)} = F(V(\mathbf{x}), L(\mathbf{y}^{(t-1)}))
$$

每次迭代，语言模型修正视觉识别的错误。

## 3. 文档理解（Document Understanding）

### 3.1 LayoutLM 架构

**核心创新**：在 BERT 基础上加入 2D 位置编码。

**输入表示**（LayoutLMv2/v3）：

$$
\mathbf{h}_i = \mathbf{e}_{token} + \mathbf{e}_{1D\_pos} + \mathbf{e}_{2D\_pos} + \mathbf{e}_{segment} + \mathbf{e}_{image}
$$

其中 2D 位置编码：

$$
\mathbf{e}_{2D\_pos} = \mathbf{E}_{x_0}(x_0) + \mathbf{E}_{y_0}(y_0) + \mathbf{E}_{x_1}(x_1) + \mathbf{E}_{y_1}(y_1) + \mathbf{E}_{w}(w) + \mathbf{E}_{h}(h)
$$

$[x_0, y_0, x_1, y_1]$ 是文字框的归一化坐标，$w, h$ 是宽高。

### 3.2 LayoutLMv3 预训练目标

**MLM（Masked Language Modeling）**：随机遮蔽 15% 的文本 token

$$
\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(t_i | \mathbf{t}_{\backslash \mathcal{M}}, \mathbf{v})
$$

**MIM（Masked Image Modeling）**：遮蔽 40% 的图像 patch

$$
\mathcal{L}_{MIM} = -\sum_{j \in \mathcal{M}'} \log P(v_j | \mathbf{v}_{\backslash \mathcal{M}'}, \mathbf{t})
$$

**WPA（Word-Patch Alignment）**：对齐文字 token 和对应图像区域

$$
\mathcal{L}_{WPA} = -\sum_{(i,j) \in \mathcal{A}} \log \sigma(\mathbf{h}_i^T \mathbf{g}_j) - \sum_{(i,k) \notin \mathcal{A}} \log(1 - \sigma(\mathbf{h}_i^T \mathbf{g}_k))
$$

### 3.3 空间感知注意力

标准注意力 + 空间偏置：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \mathbf{B}_{spatial}\right) V
$$

$$
B_{spatial}(i, j) = f(\Delta x_{ij}, \Delta y_{ij})
$$

其中 $\Delta x_{ij} = x_i - x_j$，$\Delta y_{ij} = y_i - y_j$ 是两个 token 的空间距离。

## 4. 广告文字提取

### 4.1 文字区域分类

检测到的文字区域按类型分类：

$$
\hat{y}_i = \text{softmax}(\mathbf{W}_c [\mathbf{h}_i^{visual}; \mathbf{h}_i^{text}; \mathbf{h}_i^{spatial}])
$$

类型包括：标题、促销文案、价格、品牌名、产品描述、行动号召（CTA）等。

### 4.2 结构化信息抽取

使用 BIO 标注提取关键字段：

$$
P(y_t | y_{<t}, \mathbf{x}) = \text{CRF}(\mathbf{H})
$$

CRF 建模标签间的转移概率：

$$
P(\mathbf{y} | \mathbf{x}) = \frac{\exp(\sum_{t} \phi(y_{t-1}, y_t, \mathbf{H}_t))}{\sum_{\mathbf{y}'} \exp(\sum_{t} \phi(y'_{t-1}, y'_t, \mathbf{H}_t))}
$$

## 5. 表格结构识别

### 5.1 表格检测与单元格分割

将表格识别分解为：
- **行/列线段检测**：分割网络输出横线/竖线 Mask
- **单元格合并检测**：判断相邻单元格是否合并

$$
P(\text{merge}_{i,j}) = \sigma(\mathbf{W}[\mathbf{h}_i; \mathbf{h}_j; |\mathbf{h}_i - \mathbf{h}_j|])
$$
