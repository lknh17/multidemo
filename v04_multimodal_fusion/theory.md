# 多模态融合原理：CLIP 与 Cross-Attention

## 1. CLIP 双塔架构

**CLIP** (Contrastive Language-Image Pre-training, OpenAI 2021) 是多模态领域的里程碑。

### 核心思想
用**对比学习**让图像编码器和文本编码器在同一个向量空间中对齐：
- 匹配的图文对 → embedding 距离近
- 不匹配的 → 距离远

### 架构
```
图像 → ViT → 图像 embedding (d)
                                  ↘ 余弦相似度矩阵 → InfoNCE Loss
文本 → Transformer → 文本 embedding (d)
```

### InfoNCE Loss
给定一个 batch 的 N 个图文对 `(I_i, T_i)`：

$$\mathcal{L}_{I2T} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)}$$

$$\mathcal{L}_{T2I} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(T_i, I_j)/\tau)}$$

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{I2T} + \mathcal{L}_{T2I})$$

**温度系数 τ**：控制相似度分布的锐度。τ 小 → 分布更尖锐（更关注困难样本），τ 大 → 分布更平滑。

## 2. Cross-Attention 融合

双塔模型的局限：图文只在最终 embedding 空间交互，缺少深层融合。

Cross-Attention 让两个模态在中间层深度交互：
- Query 来自模态 A，Key/Value 来自模态 B
- 模态 A 的每个 token 可以"查看"模态 B 的所有 token

## 3. 双塔 vs 融合模型

| 特性 | 双塔 (CLIP) | 融合模型 (Cross-Attn) |
|------|-------------|----------------------|
| 推理效率 | 高（可预计算 embedding） | 低（需要双模态同时输入） |
| 交互深度 | 浅（只在最终层交互） | 深（中间层交互） |
| 适用场景 | 检索、匹配 | 理解、生成 |
| 代表模型 | CLIP, ALIGN | Flamingo, Qwen-VL |

**关键洞察**：广告 embedding 场景通常需要**双塔结构**（离线计算 embedding → 在线检索），但训练时可以用 Cross-Attention 增强理解。
