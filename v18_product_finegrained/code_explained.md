# V18 - 商品理解与细粒度视觉：代码详解

## 1. 细粒度识别模块（fine_grained.py）

### 1.1 注意力裁剪

```python
class AttentionCropper(nn.Module):
    def get_attention_crop(self, attn_map, image, threshold=0.5):
        # attn_map: [B, H, W] 注意力热力图
        # 1. 二值化找高注意力区域
        mask = (attn_map > threshold * attn_map.max())

        # 2. 计算 bounding box
        rows = mask.any(dim=2)  # 哪些行有高注意力
        cols = mask.any(dim=1)  # 哪些列有高注意力
        y0, y1 = rows.nonzero().min(), rows.nonzero().max()
        x0, x1 = cols.nonzero().min(), cols.nonzero().max()

        # 3. 裁剪并 resize
        crop = image[:, :, y0:y1, x0:x1]
        crop = F.interpolate(crop, size=target_size)
        return crop
```

### 1.2 多零件检测

```python
class PartDetector(nn.Module):
    def forward(self, features, K=4):
        # features: [B, N, D] ViT patch 特征

        # 1. 学习 K 个 part query
        part_queries = self.part_queries  # [K, D]

        # 2. 每个 query 与所有 patch 计算注意力
        attn = torch.matmul(part_queries, features.T)  # [K, N]
        attn = F.softmax(attn / sqrt(D), dim=-1)

        # 3. 加权聚合得到 part 特征
        part_feats = torch.matmul(attn, features)  # [K, D]
        return part_feats, attn
```

## 2. 全局-局部联合模型

```python
class MultiGranularityModel(nn.Module):
    def forward(self, images):
        # 全局特征
        features = self.backbone(images)      # [B, 1+N, D]
        global_feat = features[:, 0]          # CLS token

        # 局部特征
        patch_feats = features[:, 1:]         # [B, N, D]
        part_feats, attn = self.part_detector(patch_feats)

        # 拼接
        combined = torch.cat([global_feat, part_feats.flatten(1)], dim=-1)

        # 全局分类 + 局部分类
        global_logits = self.global_head(global_feat)
        part_logits = [head(pf) for head, pf in zip(self.part_heads, part_feats)]

        return global_logits, part_logits, attn
```

## 3. 商品属性提取模型（model.py）

```python
class ProductAttributeModel(nn.Module):
    def forward(self, images):
        features = self.encoder(images)  # [B, D]

        # 多任务属性预测
        category = self.category_head(features)   # [B, 50]
        brand = self.brand_head(features)          # [B, 100]
        color = self.color_head(features)          # [B, 16]
        material = self.material_head(features)    # [B, 12]
        style = self.style_head(features)          # [B, 20]

        return {
            'category': category,
            'brand': brand,
            'color': torch.sigmoid(color),     # 多标签
            'material': torch.sigmoid(material),
            'style': torch.sigmoid(style),
        }
```

## 4. ArcFace 度量学习

```python
class ArcFaceHead(nn.Module):
    def forward(self, features, labels):
        # features: [B, D] 已 L2 归一化
        # 1. 计算余弦相似度
        cosine = F.linear(features, F.normalize(self.weight))

        # 2. 对正确类别加角度 margin
        theta = torch.acos(cosine.clamp(-1+1e-7, 1-1e-7))
        target_logits = torch.cos(theta[labels] + self.margin)

        # 3. 替换回 logits
        one_hot = F.one_hot(labels, self.num_classes)
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)
```

## 5. 质量评估

```python
class QualityAssessor(nn.Module):
    def forward(self, images):
        features = self.encoder(images)  # [B, D]

        # 多维度质量评分
        scores = self.quality_head(features)  # [B, 5]
        scores = torch.sigmoid(scores)  # 归一化到 [0, 1]

        # 维度: 清晰度/曝光/构图/美感/合规
        return scores
```
