# V17 - 音频理解与全模态模型：数学原理

## 1. 音频特征提取

### 1.1 Mel 频谱图

**短时傅里叶变换（STFT）**：

$$
X(m, k) = \sum_{n=0}^{N-1} x(n + m \cdot H) \cdot w(n) \cdot e^{-j2\pi kn/N}
$$

其中 $m$ 是帧索引，$k$ 是频率 bin，$H$ 是 hop length，$w(n)$ 是窗函数（Hann）。

**Mel 滤波器组**：将线性频率映射到 Mel 尺度：

$$
f_{mel} = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

**Mel 频谱**：

$$
S_{mel}(m, b) = \sum_{k} |X(m, k)|^2 \cdot H_b(k)
$$

其中 $H_b(k)$ 是第 $b$ 个三角 Mel 滤波器。通常取对数：$\log S_{mel}$。

### 1.2 MFCC（Mel 频率倒谱系数）

对 log-Mel 做 DCT（离散余弦变换）：

$$
c_n = \sum_{b=0}^{B-1} \log S_{mel}(b) \cdot \cos\left(\frac{\pi n (2b + 1)}{2B}\right)
$$

取前 13 个系数 $c_0, ..., c_{12}$ 加上 delta 和 delta-delta。

## 2. Audio Spectrogram Transformer (AST)

### 2.1 音频 Patch 化

将 Mel 频谱 $S \in \mathbb{R}^{F \times T}$ 切分为 patch：

$$
\text{Patch}_{i,j} = S[i \cdot p_f : (i+1) \cdot p_f, \; j \cdot p_t : (j+1) \cdot p_t]
$$

展平后线性投影：

$$
\mathbf{z}_{i,j} = \mathbf{W}_E \cdot \text{flatten}(\text{Patch}_{i,j}) + \mathbf{b}_E
$$

### 2.2 位置编码

二维位置编码（频率维 + 时间维）：

$$
\mathbf{p}_{i,j} = \mathbf{p}^{freq}_i + \mathbf{p}^{time}_j
$$

### 2.3 AST 编码

标准 Transformer 编码器：

$$
\mathbf{H} = \text{TransformerEncoder}([\mathbf{z}_{CLS}; \mathbf{z}_{0,0}; ...; \mathbf{z}_{F',T'}] + \mathbf{P})
$$

音频级表征使用 [CLS] token：$\mathbf{h}_{audio} = \mathbf{H}_{CLS}$

## 3. CLAP（Contrastive Language-Audio Pretraining）

### 3.1 双塔架构

- 音频塔：$f_{audio}(\mathbf{a}) = \text{Proj}_{a}(\text{AudioEncoder}(\mathbf{a}))$
- 文本塔：$f_{text}(\mathbf{t}) = \text{Proj}_{t}(\text{TextEncoder}(\mathbf{t}))$

归一化到单位球面：

$$
\hat{\mathbf{a}}_i = \frac{f_{audio}(\mathbf{a}_i)}{||f_{audio}(\mathbf{a}_i)||_2}, \quad \hat{\mathbf{t}}_j = \frac{f_{text}(\mathbf{t}_j)}{||f_{text}(\mathbf{t}_j)||_2}
$$

### 3.2 InfoNCE Loss（对称）

$$
\mathcal{L}_{a \to t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\hat{\mathbf{a}}_i^T \hat{\mathbf{t}}_i / \tau)}{\sum_{j=1}^{N} \exp(\hat{\mathbf{a}}_i^T \hat{\mathbf{t}}_j / \tau)}
$$

$$
\mathcal{L}_{CLAP} = \frac{1}{2}(\mathcal{L}_{a \to t} + \mathcal{L}_{t \to a})
$$

### 3.3 特征融合策略

全模态 CLAP 将音频加入 CLIP 框架：

$$
\mathcal{L}_{total} = \mathcal{L}_{CLIP}(I, T) + \alpha \mathcal{L}_{CLAP}(A, T) + \beta \mathcal{L}_{image-audio}(I, A)
$$

## 4. 全模态融合

### 4.1 模态对齐

三种模态映射到统一空间：

$$
\mathbf{h}_I = \text{Proj}_{I}(\text{ViT}(\mathbf{I})) \in \mathbb{R}^{N_I \times d}
$$
$$
\mathbf{h}_T = \text{Proj}_{T}(\text{LLM}(\mathbf{T})) \in \mathbb{R}^{N_T \times d}
$$
$$
\mathbf{h}_A = \text{Proj}_{A}(\text{AST}(\mathbf{A})) \in \mathbb{R}^{N_A \times d}
$$

### 4.2 Q-Former 跨模态融合

使用可学习 Query 聚合多模态信息：

$$
\mathbf{Q} = \text{LearnableQuery} \in \mathbb{R}^{M \times d}
$$

$$
\mathbf{Q}' = \text{CrossAttn}(\mathbf{Q}, [\mathbf{h}_I; \mathbf{h}_T; \mathbf{h}_A])
$$

$$
\mathbf{o} = \text{MeanPool}(\mathbf{Q}')
$$

### 4.3 模态缺失处理

使用模态嵌入标记存在/缺失：

$$
\mathbf{m}_k = \begin{cases} \mathbf{e}_{present} & \text{if modality } k \text{ available} \\ \mathbf{e}_{missing} & \text{otherwise} \end{cases}
$$

## 5. 音频事件检测

### 5.1 帧级预测

$$
P(y_t = c | \mathbf{x}) = \sigma(\mathbf{W}_c \mathbf{h}_t)
$$

### 5.2 音频标记（Audio Tagging）

$$
\hat{y}_c = \sigma\left(\frac{1}{T} \sum_{t=1}^{T} \mathbf{W}_c \mathbf{h}_t\right) \quad \text{(Mean Pooling)}
$$

或注意力池化：

$$
\hat{y}_c = \sigma\left(\sum_{t=1}^{T} \alpha_t \mathbf{W}_c \mathbf{h}_t\right), \quad \alpha_t = \frac{\exp(w^T \mathbf{h}_t)}{\sum_{t'} \exp(w^T \mathbf{h}_{t'})}
$$
