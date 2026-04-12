# V24 - 内容安全与合规：代码详解

## 1. 内容安全分类器（safety_modules.py）

### 1.1 多标签分类

```python
class ContentClassifier(nn.Module):
    def forward(self, images):
        features = self.backbone(images)  # [B, 1+N, D]
        cls_feat = features[:, 0]         # CLS token

        logits = self.classifier(cls_feat)  # [B, 8]
        probs = torch.sigmoid(logits)       # 各类独立概率

        # 与 softmax 不同，sigmoid 允许多个类别同时为 True
        # 例如一张图同时是 NSFW + Violence
        predictions = (probs > self.thresholds).float()
        return {'logits': logits, 'probs': probs, 'predictions': predictions}
```

### 1.2 阈值优化

```python
def update_thresholds(self, new_thresholds):
    # 在验证集上扫描每个类别的最优 F1 阈值
    # thresholds 不参与梯度计算
    self.thresholds.data.copy_(new_thresholds)
```

## 2. 毒性评分器（safety_modules.py）

```python
class ToxicityScorer(nn.Module):
    def forward(self, token_ids, images):
        # 1. 编码
        text_feats = self.text_encoder(token_ids)    # [B, L, D]
        image_feats = self.image_encoder(images)     # [B, 1+N, D]

        # 2. 跨模态融合（文本 query 图像）
        fused, _ = self.cross_attn(text_feats, image_feats, image_feats)
        fused = self.fusion_norm(fused + text_feats)  # 残差
        pooled = fused.mean(dim=1)                    # [B, D]

        # 3. 多维毒性评分
        dim_scores = self.toxicity_head(pooled)       # [B, 6]
        # 维度：侮辱/威胁/淫秽/歧视/仇恨/正常

        # 4. 总体毒性分（结合特征+维度分）
        overall = self.overall_head(cat([pooled, dim_scores]))
        return {'dim_scores': dim_scores, 'overall_score': overall}
```

## 3. 水印嵌入与检测（safety_modules.py）

```python
class WatermarkEmbedder(nn.Module):
    def embed(self, images, watermark):
        # 学习一个残差（模拟 DWT 中频嵌入）
        residual = self.embed_net(images)
        watermarked = images + alpha * residual
        return watermarked.clamp(0, 1)

    def detect(self, images):
        # 从图像中提取水印比特
        logits = self.detector(images)  # [B, 32]
        # logits > 0 → bit=1, logits < 0 → bit=0
        return logits

    def forward(self, images, watermark):
        watermarked = self.embed(images, watermark)
        detected = self.detect(watermarked)
        bit_acc = ((detected > 0) == (watermark > 0)).float().mean()
        return {'watermarked': watermarked, 'bit_accuracy': bit_acc}
```

## 4. 对抗攻击（safety_modules.py）

### 4.1 FGSM

```python
class AdversarialAttacker:
    def fgsm(self, model, images, labels):
        # 1. 计算损失对输入的梯度
        images_adv = images.requires_grad_(True)
        loss = loss_fn(model(images_adv), labels)
        loss.backward()

        # 2. 沿梯度符号方向加扰动
        perturbation = epsilon * images_adv.grad.sign()
        adv = (images + perturbation).clamp(0, 1)
        return adv
```

### 4.2 PGD

```python
    def pgd(self, model, images, labels):
        adv = images + random_init(epsilon)
        for step in range(attack_steps):
            # 1. 计算梯度
            loss = loss_fn(model(adv), labels)
            loss.backward()

            # 2. 沿梯度方向迈一小步
            adv = adv + step_size * adv.grad.sign()

            # 3. 投影回 ε-ball
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0, 1)
        return adv
```

## 5. 级联安全检查（model.py）

```python
class SafetyGuardModel(nn.Module):
    def forward(self, images):
        # 1. 输入清洗（高斯平滑消除对抗扰动）
        sanitized = self.sanitizer(images)

        # 2. 快速安全分类
        probs = self.fast_classifier(sanitized)['probs']
        max_prob = probs.max(dim=-1).values

        # 3. 级联决策
        # max_prob > 0.9 → 直接拒绝
        # max_prob < 0.3 → 直接通过
        # 中间 → 需要深度模型复审
        decision = torch.where(max_prob > 0.9, 1,
                    torch.where(max_prob < 0.3, 0, 2))
        return {'decision': decision, 'probs': probs}
```

## 6. Platt Scaling 校准（model.py）

```python
class CalibratedClassifier(nn.Module):
    def forward(self, images):
        raw_logits = self.base(images)['logits']

        # Platt Scaling: 学习 a * z + b
        calibrated = sigmoid(self.platt_a * raw_logits + self.platt_b)

        # Temperature Scaling: 只学一个 T
        temp_probs = sigmoid(raw_logits / self.temperature)

        return {'platt_probs': calibrated, 'temp_probs': temp_probs}

    def compute_ece(self, probs, labels, n_bins=10):
        # 分桶计算置信度-准确率差距
        for bin in bins:
            gap = |avg_confidence - avg_accuracy|
        ece = weighted_sum(gaps)
```
