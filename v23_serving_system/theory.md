# V23 - 在线推理服务系统：数学原理

## 1. 排队论与 Little's Law

### 1.1 Little's Law

系统中的平均请求数 $L$、到达速率 $\lambda$ 和平均等待时间 $W$ 之间的关系：

$$
L = \lambda W
$$

- $L$：系统中平均排队的请求数
- $\lambda$：请求到达速率（QPS）
- $W$：请求的平均响应时间（端到端延迟）

### 1.2 M/M/c 排队模型

服务系统通常用 M/M/c 排队模型描述（c 个并行 worker）：

$$
\rho = \frac{\lambda}{c \mu}
$$

其中 $\mu$ 是单个 worker 的服务速率。系统稳定条件：$\rho < 1$。

平均等待时间：

$$
W_q = \frac{C(c, \rho)}{c \mu (1 - \rho)}
$$

$C(c, \rho)$ 为 Erlang-C 公式（等待概率）。

## 2. 动态批处理

### 2.1 批处理吞吐量

单次推理延迟 $T(B)$ 与批大小 $B$ 的关系（GPU 并行性）：

$$
T(B) \approx T_0 + \alpha \cdot B
$$

- $T_0$：固定开销（kernel 启动、内存拷贝）
- $\alpha$：每增加一个样本的边际延迟

吞吐量：

$$
\text{Throughput}(B) = \frac{B}{T(B)} = \frac{B}{T_0 + \alpha B}
$$

当 $B \to \infty$ 时，吞吐量趋近 $1/\alpha$。

### 2.2 延迟-吞吐量 Tradeoff

动态批处理的超时等待窗口 $\tau$：

$$
\text{Total Latency} = \underbrace{\tau}_{\text{等待凑批}} + \underbrace{T(B)}_{\text{推理}} + \underbrace{T_{post}}_{\text{后处理}}
$$

优化目标：在 $\text{Total Latency} \leq L_{max}$ 约束下，最大化吞吐量。

## 3. ONNX 图优化

### 3.1 算子融合（Operator Fusion）

将多个连续算子合并为一个 kernel，减少内存读写：

$$
\text{MatMul} \to \text{Add} \to \text{ReLU} \quad \Rightarrow \quad \text{FusedMatMulBiasReLU}
$$

加速比：

$$
\text{Speedup} = \frac{T_{unfused}}{T_{fused}} = \frac{\sum_i (T_{compute_i} + T_{io_i})}{T_{fused\_compute} + T_{io}}
$$

### 3.2 常量折叠（Constant Folding）

编译期计算不依赖输入的子图：

$$
y = x \cdot (W_1 W_2) + b \quad \Rightarrow \quad y = x \cdot W_{merged} + b
$$

## 4. TensorRT 内核融合

### 4.1 Layer Fusion

垂直融合（串行层）+ 水平融合（并行层）：

$$
\text{Conv} \to \text{BN} \to \text{ReLU} \quad \Rightarrow \quad \text{CBR\_Fused}
$$

### 4.2 精度校准

INT8 量化的缩放因子：

$$
x_{int8} = \text{round}\left(\frac{x_{fp32}}{s}\right), \quad s = \frac{\max(|x|)}{127}
$$

KL 散度校准：

$$
s^* = \arg\min_s \text{KL}(P_{fp32} \| Q_{int8}(s))
$$

## 5. ANN 搜索复杂度

### 5.1 暴力搜索（Flat）

$$
O(n \cdot d)
$$

$n$ 为向量总数，$d$ 为维度。

### 5.2 IVF（Inverted File Index）

训练阶段：K-Means 聚类得到 $nlist$ 个聚类中心。

搜索阶段：仅搜索最近的 $nprobe$ 个簇：

$$
O\left(nprobe \cdot \frac{n}{nlist} \cdot d\right)
$$

召回率与 $nprobe/nlist$ 比值正相关。

### 5.3 HNSW（Hierarchical Navigable Small World）

$$
O(d \cdot \log n)
$$

## 6. 缓存命中率建模

### 6.1 LRU 缓存命中率

假设请求服从 Zipf 分布（参数 $\alpha$），缓存大小 $C$，总条目 $N$：

$$
\text{Hit Rate} \approx \frac{\sum_{i=1}^{C} i^{-\alpha}}{\sum_{i=1}^{N} i^{-\alpha}}
$$

当 $\alpha > 1$ 时，少量热门条目贡献大部分请求，缓存效果好。

### 6.2 TTL 失效模型

缓存条目的有效概率：

$$
P(\text{valid}) = e^{-t/\text{TTL}}
$$

有效命中率 = 命中率 × 有效概率。

## 7. 负载均衡策略

### 7.1 Round-Robin

请求均匀分配到 $N$ 个节点：

$$
\text{node}(r) = r \mod N
$$

### 7.2 Least-Connection

选择当前连接数最少的节点：

$$
\text{node}^* = \arg\min_i \text{active\_connections}_i
$$

### 7.3 Consistent Hashing

将请求 key 和节点都映射到哈希环上：

$$
h(key) \to [0, 2^{32})
$$

顺时针查找最近的节点。添加/移除节点时仅影响相邻区间，迁移代价 $O(K/N)$。

## 8. 量化精度分析

### 8.1 量化误差

$$
\epsilon = x_{fp32} - s \cdot x_{int8} = x_{fp32} - s \cdot \text{round}(x_{fp32}/s)
$$

$$
|\epsilon| \leq \frac{s}{2}
$$

### 8.2 精度-速度 Tradeoff

$$
\text{Speedup}_{INT8} \approx 2\text{-}4\times \quad \text{vs FP32}
$$

$$
\text{Accuracy Drop} \approx 0.1\%\text{-}2\% \quad \text{(经校准后)}
$$
