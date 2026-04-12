# V16 - OCR 与文档理解：代码详解

## 1. OCR 检测模块（ocr_modules.py）

### 1.1 DBNet 文字检测

核心是可微二值化：

```python
class DifferentiableBinarization(nn.Module):
    def forward(self, prob_map, thresh_map, k=50):
        # 可微二值化：替代固定阈值
        # k 越大越接近硬二值化，但保持可微
        binary = torch.sigmoid(k * (prob_map - thresh_map))
        return binary
```

**为什么 k=50 有效**：当 $p - t = 0.1$ 时，$\sigma(50 \times 0.1) = 0.993$，已接近 1。使得决策边界清晰但仍可传梯度。

### 1.2 FPN 特征融合

检测需要多尺度特征来处理不同大小的文字：

```python
# 构建自底向上特征金字塔
c2 = self.layer2(c1)  # 1/4
c3 = self.layer3(c2)  # 1/8
c4 = self.layer4(c3)  # 1/16

# 自顶向下融合
p4 = self.lateral4(c4)
p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:])
p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:])

# 拼接多尺度特征
fused = torch.cat([
    F.interpolate(p, size=p2.shape[2:]) for p in [p2, p3, p4]
], dim=1)  # [B, 3C, H/4, W/4]
```

## 2. 文字识别（CRNN 风格 + Attention）

### 2.1 CNN 特征提取

```python
# 输入: [B, 1, 32, 100] 灰度文字图像
# 经过多层 CNN 后: [B, D, 1, W'] -> [B, D, W']
# W' 对应文字序列的时间步

features = self.cnn(image)  # [B, D, 1, W]
features = features.squeeze(2)  # [B, D, W]
features = features.permute(0, 2, 1)  # [B, W, D] -> 序列形式
```

### 2.2 CTC 解码

```python
# CTC Loss：不需要精确对齐，允许重复和空白
logits = self.classifier(features)  # [B, W, vocab_size+1]
log_probs = F.log_softmax(logits, dim=-1)

# CTC Loss 需要 (T, B, C) 格式
loss = F.ctc_loss(
    log_probs.permute(1, 0, 2),  # [T, B, C]
    targets,
    input_lengths,
    target_lengths,
    blank=0,
)

# 贪心解码
preds = logits.argmax(dim=-1)  # [B, T]
# 去重复 + 去空白
decoded = ctc_greedy_decode(preds, blank_id=0)
```

### 2.3 Attention 解码器（替代 CTC）

```python
# Attention 解码器可以建模字符间依赖
for t in range(max_len):
    # 输入：上一步预测的字符
    embed = self.char_embed(prev_token)

    # Cross-attention with CNN features
    context, attn_weights = self.attention(embed, features)

    # 预测下一个字符
    logits = self.classifier(context)
    next_char = logits.argmax(-1)
    prev_token = next_char
```

优势对比：
- CTC：训练快，但假设帧独立
- Attention：建模字符依赖，准确率更高，但慢

## 3. LayoutLM 文档理解（model.py）

### 3.1 2D 位置编码实现

```python
class Layout2DPositionEmbedding(nn.Module):
    def __init__(self, config):
        # 6 个独立的 Embedding 表
        self.x0_embed = nn.Embedding(config.max_position_x, config.d_model)
        self.y0_embed = nn.Embedding(config.max_position_y, config.d_model)
        self.x1_embed = nn.Embedding(config.max_position_x, config.d_model)
        self.y1_embed = nn.Embedding(config.max_position_y, config.d_model)
        self.w_embed = nn.Embedding(config.max_width, config.d_model)
        self.h_embed = nn.Embedding(config.max_height, config.d_model)

    def forward(self, bbox):
        # bbox: [B, L, 4] -> (x0, y0, x1, y1)
        x0, y0, x1, y1 = bbox.unbind(-1)
        w = (x1 - x0).clamp(0, self.max_w - 1)
        h = (y1 - y0).clamp(0, self.max_h - 1)

        return (self.x0_embed(x0) + self.y0_embed(y0) +
                self.x1_embed(x1) + self.y1_embed(y1) +
                self.w_embed(w) + self.h_embed(h))
```

**关键理解**：6 个 Embedding 表独立训练，模型学习到"同一行的 token y 坐标相似"、"同一列的 token x 坐标相似"等空间关系。

### 3.2 多模态输入融合

```python
# 文本分支
text_embed = self.token_embed(token_ids)       # [B, L, D]
text_embed += self.position_embed(pos_ids)     # 1D 位置
text_embed += self.layout_embed(bboxes)        # 2D 位置

# 图像分支
image_embed = self.patch_embed(images)         # [B, N, D]
image_embed += self.image_pos_embed(img_pos)   # 图像位置

# 拼接
combined = torch.cat([text_embed, image_embed], dim=1)  # [B, L+N, D]

# Transformer 编码
output = self.transformer(combined)
text_output = output[:, :L]      # 文本输出
image_output = output[:, L:]     # 图像输出
```

### 3.3 MLM 预训练

```python
def create_mlm_inputs(token_ids, mlm_prob=0.15):
    """创建 MLM 输入：15% 随机遮蔽"""
    labels = token_ids.clone()
    mask = torch.rand_like(token_ids.float()) < mlm_prob

    # 80% 替换为 [MASK]
    replace_mask = mask & (torch.rand_like(mask.float()) < 0.8)
    token_ids[replace_mask] = MASK_TOKEN_ID

    # 10% 随机替换
    random_mask = mask & ~replace_mask & (torch.rand_like(mask.float()) < 0.5)
    token_ids[random_mask] = torch.randint_like(token_ids[random_mask], 3, vocab_size)

    # 10% 保持不变
    labels[~mask] = -100  # ignore in loss
    return token_ids, labels
```

## 4. 广告文字提取

### 4.1 多类型文字分类

```python
class AdTextClassifier(nn.Module):
    """对检测到的文字区域进行类型分类"""

    def forward(self, visual_feat, text_feat, spatial_feat):
        # 三种特征拼接
        combined = torch.cat([visual_feat, text_feat, spatial_feat], dim=-1)

        # MLP 分类
        logits = self.classifier(combined)  # [B, N_regions, num_types]
        return logits

# 文字类型定义
TEXT_TYPES = {
    0: '标题',
    1: '促销文案',
    2: '价格',
    3: '品牌名',
    4: '产品描述',
    5: '行动号召(CTA)',
    6: '免责声明',
    7: '其他',
}
```

### 4.2 空间关系建模

广告中文字区域的空间关系很重要：

```python
def compute_spatial_features(bboxes):
    """计算文字区域间的空间关系特征"""
    N = bboxes.shape[1]

    # 两两计算相对位置
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2  # 中心 x
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2  # 中心 y

    # 相对距离
    dx = cx.unsqueeze(-1) - cx.unsqueeze(-2)  # [B, N, N]
    dy = cy.unsqueeze(-1) - cy.unsqueeze(-2)

    # 是否同行/同列
    same_row = (dy.abs() < threshold).float()
    same_col = (dx.abs() < threshold).float()

    return dx, dy, same_row, same_col
```

## 5. 端到端流水线

```
输入图像
   ↓
[文字检测 DBNet] → 文字区域 bounding boxes
   ↓
[文字识别 CRNN] → 每个区域的文字内容
   ↓
[LayoutLM 编码] → 文本 + 布局 + 图像 融合特征
   ↓
[下游任务头]
  ├── 文字类型分类 → 标题/价格/促销/...
  ├── 关键信息抽取 → BIO 序列标注
  └── 文档分类    → 广告类型/风格
```
