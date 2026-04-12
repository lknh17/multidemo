# 文本编码器与 LLM 原理

## 1. 自回归语言模型

GPT 式模型的核心：给定前面的 token，预测下一个 token。

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})$$

### 因果注意力掩码
只能看到当前位置之前的 token，通过下三角掩码实现。这是 Encoder（BERT）和 Decoder（GPT）的核心区别。

## 2. RoPE 旋转位置编码

当前大模型主流选择（Qwen、LLaMA 都用 RoPE）。

核心思想：对 Q 和 K 施加与位置相关的旋转，使得 `Q_m · K_n` 的内积只依赖于相对位置 `m-n`。

对于 2D 向量对：
$$f(q, m) = q \cdot e^{im\theta} = \begin{pmatrix} q_0\cos(m\theta) - q_1\sin(m\theta) \\ q_0\sin(m\theta) + q_1\cos(m\theta) \end{pmatrix}$$

## 3. KV Cache

推理优化的核心技术。自回归生成时，每步只新增一个 token，但朴素实现需要重新计算所有前缀的 K 和 V。

KV Cache：缓存历史步骤的 K、V，每步只计算新 token 的 K、V 并追加。

- 无 Cache：每步计算量 O(n²d) → 总计 O(n³d)
- 有 Cache：每步计算量 O(nd) → 总计 O(n²d)

## 4. 采样策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| Greedy | 取概率最高的 | 确定性任务 |
| Top-k | 只从概率最高的 k 个中采样 | 通用生成 |
| Top-p (Nucleus) | 从累积概率达到 p 的最小集合中采样 | 更灵活 |
| Temperature | logits / T，T>1 更随机，T<1 更确定 | 控制随机度 |

## 5. RMSNorm

LLaMA/Qwen 使用的归一化方式，比 LayerNorm 更轻量（去掉均值中心化）。
