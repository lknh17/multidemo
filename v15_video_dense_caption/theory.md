# V15 - 视频理解与 Dense Captioning：数学原理

## 1. 视频时序建模基础

### 1.1 3D 卷积

将 2D 卷积扩展到时序维度：

$$
y_{t,i,j} = \sum_{\tau=0}^{T_k-1} \sum_{m=0}^{H_k-1} \sum_{n=0}^{W_k-1} w_{\tau,m,n} \cdot x_{t+\tau, i+m, j+n}
$$

其中 $T_k$ 是时序核大小。**核心问题**：计算量 $O(T_k \cdot H_k \cdot W_k \cdot C_{in} \cdot C_{out})$，时空耦合导致参数量爆炸。

**R(2+1)D 分解**：将 3D 卷积分解为空间 2D + 时序 1D：

$$
\text{Conv3D}(T_k, H_k, W_k) \approx \text{Conv2D}(1, H_k, W_k) \circ \text{Conv1D}(T_k, 1, 1)
$$

参数量从 $T_k H_k W_k C^2$ 降至 $H_k W_k C^2 + T_k C^2$。

### 1.2 TimeSformer：分离时空注意力

**核心思想**：将视频 Token 的自注意力分解为空间注意力 + 时序注意力。

给定视频 $\mathbf{X} \in \mathbb{R}^{T \times N \times D}$（$T$ 帧，每帧 $N$ 个 patch），Divided Space-Time Attention：

**Step 1 - 时序注意力**：对每个空间位置 $n$，在 $T$ 帧之间做注意力：

$$
\mathbf{Z}^{(t)}_{:,n,:} = \text{Attn}(\mathbf{X}_{:,n,:} \mathbf{W}_Q, \mathbf{X}_{:,n,:} \mathbf{W}_K, \mathbf{X}_{:,n,:} \mathbf{W}_V)
$$

计算复杂度：$O(N \cdot T^2 \cdot D)$

**Step 2 - 空间注意力**：对每帧 $t$，在 $N$ 个 patch 之间做注意力：

$$
\mathbf{Y}_{t,:,:} = \text{Attn}(\mathbf{Z}^{(t)}_{t,:,:} \mathbf{W}'_Q, \mathbf{Z}^{(t)}_{t,:,:} \mathbf{W}'_K, \mathbf{Z}^{(t)}_{t,:,:} \mathbf{W}'_V)
$$

计算复杂度：$O(T \cdot N^2 \cdot D)$

**总复杂度**：$O(N T^2 D + T N^2 D)$，远小于联合注意力的 $O((NT)^2 D)$。

### 1.3 Video Swin Transformer

将 Swin Transformer 的局部窗口注意力扩展到 3D：

- **3D 窗口划分**：窗口大小 $(T_w, H_w, W_w)$，窗口内自注意力
- **3D Shifted Window**：偏移 $(T_w/2, H_w/2, W_w/2)$ 实现跨窗口交互
- **相对位置偏置**：$B \in \mathbb{R}^{(2T_w-1) \times (2H_w-1) \times (2W_w-1)}$

## 2. Dense Video Captioning

### 2.1 问题定义

给定视频 $\mathbf{V}$，输出一组事件集合 $\{(s_i, e_i, \mathbf{c}_i)\}_{i=1}^{M}$：
- $(s_i, e_i)$：第 $i$ 个事件的起止时间
- $\mathbf{c}_i$：该事件的自然语言描述

### 2.2 时序提议网络（Temporal Proposal Network）

**Anchor-based 方法**：在每个时间步 $t$ 预设多个 anchor $(t, l_a)$（中心 $t$，长度 $l_a$）：

$$
p_t^a = \sigma(\mathbf{w}_{cls}^T \mathbf{h}_t + b_{cls})
$$

$$
(\Delta c_t^a, \Delta l_t^a) = \mathbf{W}_{reg} \mathbf{h}_t + \mathbf{b}_{reg}
$$

预测中心偏移 $\Delta c$ 和长度缩放 $\Delta l$，最终提议：

$$
\hat{s}_i = t + \Delta c - \frac{l_a \cdot e^{\Delta l}}{2}, \quad \hat{e}_i = t + \Delta c + \frac{l_a \cdot e^{\Delta l}}{2}
$$

**Anchor-free 方法（PDVC 风格）**：使用 Learnable Event Query：

$$
\mathbf{Q}_{event} \in \mathbb{R}^{N_q \times D}
$$

通过 Cross-Attention 与视频特征交互：

$$
\hat{\mathbf{Q}} = \text{CrossAttn}(\mathbf{Q}_{event}, \mathbf{H}_{video}, \mathbf{H}_{video})
$$

每个 Query 回归 $(s_i, e_i)$ 和分类置信度 $p_i$。

### 2.3 时序 NMS（Non-Maximum Suppression）

计算时序 IoU：

$$
\text{tIoU}(A, B) = \frac{\max(0, \min(e_A, e_B) - \max(s_A, s_B))}{\max(e_A, e_B) - \min(s_A, s_B)}
$$

**Soft-NMS**：不直接删除，而是衰减置信度：

$$
p_i \leftarrow p_i \cdot e^{-\frac{\text{tIoU}(M, B_i)^2}{\sigma}}
$$

其中 $M$ 是当前最高置信度提议。优势：避免密集事件被误删。

### 2.4 描述生成器

**Proposal-conditioned Caption**：给定提议 $(s_i, e_i)$，提取局部特征：

$$
\mathbf{h}_i^{local} = \text{RoIPool}(\mathbf{H}_{video}, s_i, e_i)
$$

使用自回归解码器生成描述：

$$
P(\mathbf{c}_i | \mathbf{V}, s_i, e_i) = \prod_{j=1}^{|\mathbf{c}_i|} P(c_{i,j} | c_{i,<j}, \mathbf{h}_i^{local})
$$

### 2.5 端到端联合训练损失

$$
\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{reg} \mathcal{L}_{reg} + \lambda_{cap} \mathcal{L}_{cap}
$$

- $\mathcal{L}_{cls}$：Focal Loss，处理正负样本不均衡
- $\mathcal{L}_{reg}$：GIoU Loss + L1 Loss 的组合
- $\mathcal{L}_{cap}$：Cross-Entropy（Teacher Forcing）

**匈牙利匹配（Hungarian Matching）**：

对于 DETR 风格的端到端方法，需要预测和 GT 之间的最优二部匹配：

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})
$$

$$
\mathcal{L}_{match} = -\mathbb{1}_{c_i \neq \varnothing} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing} \mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})
$$

## 3. Deformable 时序注意力

### 3.1 标准 vs Deformable

标准注意力在所有时间步上计算，复杂度 $O(T^2)$。Deformable Attention 只关注 $K$ 个关键采样点：

$$
\text{DeformAttn}(\mathbf{q}, \hat{t}, \mathbf{H}) = \sum_{k=1}^{K} A_k \cdot \mathbf{W}_v \mathbf{H}(\hat{t} + \Delta t_k)
$$

- $\hat{t}$：参考时间点
- $\Delta t_k$：可学习的偏移量，$\Delta t_k = f_{\text{offset}}(\mathbf{q})$
- $A_k$：注意力权重，$A_k = \text{softmax}(f_{\text{attn}}(\mathbf{q}))_k$
- $\mathbf{H}(\hat{t} + \Delta t_k)$：双线性插值获取非整数位置的特征

复杂度降为 $O(T \cdot K)$，$K \ll T$。

### 3.2 多尺度 Deformable 时序注意力

在 $L$ 个时间尺度上采样：

$$
\text{MSDeformAttn}(\mathbf{q}, \hat{t}, \{\mathbf{H}^l\}_{l=1}^L) = \sum_{l=1}^{L} \sum_{k=1}^{K} A_{lk} \cdot \mathbf{W}_v \mathbf{H}^l(\hat{t}_l + \Delta t_{lk})
$$

不同尺度捕捉不同粒度的时序模式：细粒度动作 vs 长程事件。

## 4. 时序 Grounding

### 4.1 Moment Retrieval

给定文本查询 $\mathbf{q}$，在视频中定位对应时间段 $(s, e)$：

$$
(\hat{s}, \hat{e}) = \arg\max_{(s,e)} f(\mathbf{V}_{s:e}, \mathbf{q})
$$

**Proposal-based**：先生成候选时段，再排序。

**Proposal-free（回归式）**：直接预测 $(s, e)$：

$$
\mathbf{h}_{fused} = \text{CrossAttn}(\mathbf{H}_{text}, \mathbf{H}_{video})
$$

$$
(\hat{s}, \hat{e}) = \text{MLP}(\text{Pool}(\mathbf{h}_{fused}))
$$

### 4.2 Span Loss

$$
\mathcal{L}_{span} = \lambda_1 \|(\hat{s}, \hat{e}) - (s^*, e^*)\|_1 + \lambda_2 \mathcal{L}_{gIoU}((\hat{s}, \hat{e}), (s^*, e^*))
$$

gIoU Loss 定义（时序版本）：

$$
\mathcal{L}_{gIoU} = 1 - \left( \text{tIoU} - \frac{|\text{Hull} \setminus \text{Union}|}{|\text{Hull}|} \right)
$$

## 5. 长视频优化策略

### 5.1 Token Merging（ToMe for Video）

相似 Token 合并减少序列长度：

$$
\text{sim}(i, j) = \frac{\mathbf{h}_i^T \mathbf{h}_j}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|}
$$

通过二部匹配（Bipartite Matching）选择 Top-$r$ 对最相似 Token 合并：

$$
\mathbf{h}_{merged} = \frac{n_i \mathbf{h}_i + n_j \mathbf{h}_j}{n_i + n_j}
$$

每层合并 $r$ 对，$L$ 层后序列长度从 $N$ 减少到约 $N - L \cdot r$。

### 5.2 关键帧采样策略

**均匀采样**：$t_k = k \cdot T / N_f$

**基于运动的采样**：计算帧间光流差异：

$$
\text{motion}(t) = \|\mathbf{F}_{t \to t+1}\|_2
$$

选择运动变化最大的帧。

**基于聚类的采样**：K-Means 聚类帧特征，每簇选最近中心帧。

### 5.3 分层时序聚合

构建时序金字塔 $\{\mathbf{H}^1, \mathbf{H}^2, ..., \mathbf{H}^L\}$：

$$
\mathbf{H}^{l+1} = \text{AvgPool1D}(\mathbf{H}^l, \text{stride}=2)
$$

FPN 风格自顶向下融合：

$$
\mathbf{H}^l_{fused} = \mathbf{H}^l + \text{Upsample}(\mathbf{H}^{l+1}_{fused})
$$

不同层级检测不同时长的事件。
