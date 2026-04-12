# p08 原理详解 - 推理部署

> 本文档详细讲解 LLM 推理部署的理论基础：自回归推理流程、KV Cache、PagedAttention、Continuous Batching、Speculative Decoding、性能指标、框架对比与最佳实践。

---

## 1. 自回归推理流程

### Prefill 与 Decode 两阶段

LLM 推理分为两个截然不同的阶段：

**Prefill（预填充）阶段**：
- 输入：用户的完整 prompt（如 128 个 token）
- 计算：对所有 token **并行**计算注意力，生成 KV Cache
- 特点：**计算密集型**（compute-bound），充分利用 GPU 的并行计算能力
- 耗时：与 prompt 长度成正比，O(n²) 的注意力计算

**Decode（解码）阶段**：
- 输入：上一步生成的 token（单个 token）
- 计算：与已有 KV Cache 做注意力，生成下一个 token
- 特点：**访存密集型**（memory-bound），每步只生成 1 个 token
- 耗时：每步需要读取整个 KV Cache，但计算量很小

$$\text{Prefill}: \mathbf{Q} = [q_1, q_2, ..., q_n], \quad O(n^2 \cdot d)$$
$$\text{Decode}: \mathbf{q} = q_{t}, \quad O(t \cdot d) \text{ 每步}$$

### 为什么 Decode 是瓶颈

以 7B 模型为例（32 层，32 头，head_dim=128）：
- 每步 decode 需要读取的 KV Cache 大小：$2 \times 32 \times 32 \times 128 \times 2\text{bytes} \times t$ ≈ $0.5\text{MB} \times t$
- 当 t=2048 时，每步需读取约 1GB 数据
- A100 的显存带宽约 2TB/s，因此理论上限约 2000 tokens/s
- 但 FLOPs 利用率不到 1%（大量时间在"搬数据"）

这就是为什么推理优化的核心在于**减少显存读写**和**增大批处理**。

---

## 2. KV Cache 详解

### 什么是 KV Cache

在自注意力中，每个 token 需要与所有之前的 token 计算注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

如果不缓存，生成第 t 个 token 时需要重新计算前 t-1 个 token 的 K 和 V，造成大量重复计算。

**KV Cache** 将之前所有 token 的 K 和 V 向量缓存在 GPU 显存中：
- 生成第 t 个 token 时，只需计算 $q_t$ 的 K 和 V，然后与缓存的 $K_{1:t-1}$ 和 $V_{1:t-1}$ 做注意力
- 将 O(t²) 的重复计算降为 O(t)

### KV Cache 显存开销

单个 token 的 KV Cache 大小：

$$\text{KV\_per\_token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{dtype\_size}$$

| 模型 | 参数量 | KV/token (fp16) | 2048 token 的 KV Cache |
|------|--------|-----------------|----------------------|
| 0.5B | 24层, 16头, 64d | 96 KB | 192 MB |
| 7B | 32层, 32头, 128d | 512 KB | 1 GB |
| 70B | 80层, 64头, 128d | 2.5 MB | 5 GB |

对于 70B 模型，10 个并发请求的 KV Cache 就需要 50GB 显存！这就是 KV Cache 管理至关重要的原因。

### KV Cache 的浪费问题

传统实现预分配 `max_seq_len` 大小的连续显存：
- 请求可能提前结束，预分配的空间浪费
- 不同请求的长度不同，无法有效共享
- 显存碎片化严重

这正是 PagedAttention 要解决的问题。

---

## 3. PagedAttention

### 核心思想

PagedAttention 借鉴了操作系统的**虚拟内存分页**机制：

1. **物理显存**被划分为固定大小的**块（Block）**，每块存储若干 token 的 KV Cache
2. **逻辑序列**通过**页表**映射到物理块
3. 块按需分配，不预分配整个 max_seq_len
4. 请求结束后，块被回收到空闲池

### 与传统方法对比

**传统方法**：
```
请求1: [████████████________] 预分配 2048，实际用 1200
请求2: [██████______________] 预分配 2048，实际用 600
空闲:  [____________________] 碎片无法利用
```

**PagedAttention**：
```
请求1: [Block0][Block1][Block2]  按需分配 3 个块
请求2: [Block3][Block4]          按需分配 2 个块
空闲池: [Block5][Block6]...[BlockN]  统一管理
```

### 显存利用率提升

| 方法 | 显存利用率 | 并发能力 |
|------|-----------|---------|
| 传统预分配 | ~50-60% | 低（碎片浪费） |
| PagedAttention | ~95%+ | 高（按需分配） |

### 前缀共享（Prefix Sharing）

PagedAttention 还支持**前缀共享**：多个请求如果有相同的 system prompt，它们的 KV Cache 前缀可以共享同一组物理块，进一步节省显存。

$$\text{显存节省} = (N_{\text{requests}} - 1) \times L_{\text{prefix}} \times \text{KV\_per\_token}$$

---

## 4. Continuous Batching

### Static Batching 的问题

传统 static batching：
- 一批请求必须等**所有请求**都完成才能开始下一批
- 短请求被长请求"拖累"
- GPU 利用率低（短请求完成后空闲等待）

```
时间 → →
请求A: [=======]_________  完成后等待
请求B: [===============]  最长的拖住整批
请求C: [====]____________  完成后空闲
         ↑ 同时开始      ↑ 同时结束
```

### Continuous Batching（连续批处理）

Continuous batching 在**每个 decode step** 级别调度：
- 短请求完成后立即移出，空位立即插入新请求
- GPU 始终在处理最大并发数的请求
- 显著提高吞吐量

```
时间 → →
请求A: [=======]
请求B: [===============]
请求C: [====]
请求D:      [=========]     ← C 完成后立即插入
请求E:          [======]    ← A 完成后立即插入
```

### 关键参数

- **max_num_seqs**: 最大同时处理的序列数（决定了最大批大小）
- **max_num_batched_tokens**: 每步处理的最大 token 数（prefill + decode 总和）
- **chunked_prefill**: 将长 prompt 的 prefill 分块，与 decode 请求混合处理，避免长 prompt 阻塞

---

## 5. Speculative Decoding（投机解码）

### 核心思想

用一个**小模型（draft model）**快速生成多个 token 候选，然后用**大模型（target model）**一次性验证，接受正确的、拒绝错误的。

**关键洞察**: 大模型**验证** N 个 token 的成本 ≈ 生成 1 个 token 的成本（因为可以并行）。

### 工作流程

1. **Draft 阶段**: 小模型（如 0.5B）自回归生成 K 个 token: $[t_1, t_2, ..., t_K]$
2. **Verify 阶段**: 大模型（如 7B）并行验证这 K 个 token
3. **Accept/Reject**: 
   - 如果 $t_i$ 通过验证（概率满足条件），接受
   - 遇到第一个被拒绝的 $t_j$，丢弃 $t_j$ 及后续所有
   - 大模型重新采样一个 token 替换 $t_j$

### 加速比

理想情况下，每步可以接受 2-4 个 token，加速比 2-4x。

接受率取决于：
- draft model 与 target model 的相似度
- 生成内容的确定性（高确定性场景接受率更高）

$$\text{加速比} \approx \frac{1 + \alpha \cdot K}{1 + \beta}$$

其中 $\alpha$ 是平均接受率，$K$ 是投机长度，$\beta$ 是 draft/target 速度比。

### 使用条件

- Draft model 必须与 target model 使用相同的 tokenizer
- Draft model 越小越快，但接受率越低（需要平衡）
- 在 batch size 较大时加速效果减弱（GPU 已经饱和）

---

## 6. 性能指标详解

### TTFT (Time To First Token)

- **定义**: 从发送请求到收到第一个 token 的时间
- **影响因素**: prefill 计算量（prompt 长度²）、排队等待时间
- **优化**: chunked prefill、前缀缓存

$$\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}}$$

### TPS (Tokens Per Second)

- **定义**: 单个请求每秒生成的 token 数
- **影响因素**: 模型大小、批大小、显存带宽
- **计算**: $\text{TPS} = \text{output\_tokens} / \text{total\_time}$

### ITL (Inter-Token Latency)

- **定义**: 相邻两个 token 之间的时间间隔
- **与 TPS 的关系**: $\text{ITL} \approx 1 / \text{TPS}$
- **影响因素**: decode 时的 KV Cache 读取速度

### Throughput（吞吐量）

- **定义**: 系统每秒处理的请求数或 token 数
- **两种度量**:
  - 请求吞吐: requests/second
  - Token 吞吐: tokens/second（更常用）
- **影响因素**: 批大小、GPU 利用率、调度效率

### 延迟百分位 (Percentile Latency)

| 百分位 | 含义 | 关注点 |
|--------|------|--------|
| P50 | 50% 请求在此时间内完成 | 中位数体验 |
| P90 | 90% 请求在此时间内完成 | 大多数用户体验 |
| P99 | 99% 请求在此时间内完成 | 尾部延迟（SLA 关键） |

**吞吐 vs 延迟 tradeoff**: 增大批大小提高吞吐，但也会增大单请求延迟。

---

## 7. 框架对比

### vLLM

- **核心**: PagedAttention + Continuous Batching
- **优势**: 最成熟的高性能推理引擎，社区活跃
- **劣势**: 启动较慢，不支持复杂的 LLM 程序编排
- **适用**: 生产环境高吞吐服务

### SGLang

- **核心**: RadixAttention（基数树缓存）
- **优势**: 自动前缀复用，适合多轮对话和复杂 LLM 程序
- **RadixAttention 原理**: 用基数树索引所有活跃请求的 KV Cache 前缀，新请求自动匹配最长公共前缀，复用已有 KV Cache
- **适用**: 多轮对话、few-shot、复杂推理链

### Ollama

- **核心**: llama.cpp（CPU/GPU 混合推理）
- **优势**: 极简部署，支持 GGUF 量化，可在纯 CPU 运行
- **劣势**: 吞吐远低于 vLLM/SGLang
- **适用**: 本地开发、边缘设备、离线场景

### 性能对比（7B 模型，单卡 A100）

| 指标 | vLLM | SGLang | Ollama |
|------|------|--------|--------|
| TTFT (单请求) | ~50ms | ~45ms | ~200ms |
| TPS (单请求) | ~80 | ~85 | ~30 |
| 吞吐 (32并发) | ~1500 tok/s | ~1600 tok/s | ~100 tok/s |
| 显存效率 | ★★★★★ | ★★★★★ | ★★★ |
| 前缀复用 | 手动启用 | 自动 | 有限 |

---

## 8. 部署最佳实践

### 模型选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 对话机器人 | Instruct 模型 + vLLM | 高吞吐 + 多轮优化 |
| 代码补全 | Code 模型 + SGLang | 前缀复用（代码上下文相似） |
| 离线批处理 | Base 模型 + vLLM | 最大吞吐 |
| 边缘设备 | GGUF 量化 + Ollama | 低资源要求 |

### 显存规划

$$\text{总显存} = \text{模型参数} + \text{KV Cache} + \text{激活值} + \text{框架开销}$$

| 模型 | 参数 (fp16) | 推荐显存 (高并发) |
|------|------------|------------------|
| 0.5B | ~1 GB | 4-8 GB |
| 7B | ~14 GB | 24-40 GB |
| 13B | ~26 GB | 40-80 GB |
| 70B | ~140 GB | 4×A100 或 8×A100 |

### 服务架构

```
                    ┌──── vLLM Instance 0 (GPU 0)
Load Balancer ──────┼──── vLLM Instance 1 (GPU 1)
(Nginx/HAProxy)     └──── vLLM Instance 2 (GPU 2)
```

- **多实例负载均衡**: 对无状态请求最简单有效
- **Tensor Parallel**: 单个大模型跨多卡
- **Data Parallel**: 多个模型副本处理不同请求

### 监控指标

| 指标 | 告警阈值 | 含义 |
|------|---------|------|
| P99 延迟 | >5s | 用户体验恶化 |
| GPU 利用率 | <30% | 资源浪费 |
| 队列长度 | >100 | 需要扩容 |
| OOM 次数 | >0 | 降低并发或显存配置 |
| 错误率 | >1% | 服务异常 |

### 成本优化

1. **按需扩缩容**: 低峰期减少 GPU 实例
2. **模型量化**: AWQ 量化可降低 50% 显存，吞吐反而提高
3. **Speculative Decoding**: 用 0.5B 辅助 7B，可提升 2-3x 速度
4. **KV Cache 压缩**: 使用 GQA/MQA 减少 KV Head 数量
5. **Prefix Caching**: 系统 prompt 缓存避免重复计算
