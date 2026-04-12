# V17 - 音频理解与全模态模型：代码详解

## 1. 音频特征提取（audio_modules.py）

### 1.1 Mel 频谱图计算

```python
class MelSpectrogram(nn.Module):
    def forward(self, waveform):
        # waveform: [B, T] 原始音频波形
        # 1. STFT
        stft = torch.stft(waveform, n_fft, hop_length, return_complex=True)
        # stft: [B, n_fft//2+1, T'] 复数频谱

        # 2. 取功率谱
        power = stft.abs() ** 2  # [B, F, T']

        # 3. Mel 滤波
        mel = torch.matmul(self.mel_filters, power)  # [B, n_mels, T']

        # 4. 取对数
        log_mel = torch.log(mel + 1e-9)  # [B, n_mels, T']
        return log_mel
```

**关键理解**：Mel 滤波器将 F 个线性频率 bin 压缩到 n_mels 个对数分布的频段，模拟人耳对音高的感知。

### 1.2 AST 音频 Patch 化

```python
class AudioPatchEmbed(nn.Module):
    def __init__(self, n_mels, time_patch, freq_patch, d_model):
        # 使用 2D 卷积实现 patch embedding
        self.proj = nn.Conv2d(
            1, d_model,
            kernel_size=(freq_patch, time_patch),
            stride=(freq_patch, time_patch)
        )

    def forward(self, mel):
        # mel: [B, n_mels, T] -> [B, 1, n_mels, T]
        x = mel.unsqueeze(1)
        x = self.proj(x)       # [B, D, n_freq_patches, n_time_patches]
        x = x.flatten(2)       # [B, D, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, D]
        return x
```

## 2. AST 编码器

```python
class AudioSpectrogramTransformer(nn.Module):
    def forward(self, mel_spec):
        # 1. Patch Embedding
        patches = self.patch_embed(mel_spec)   # [B, N, D]

        # 2. 加入 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, patches], dim=1)    # [B, 1+N, D]

        # 3. 加位置编码
        x = x + self.pos_embed[:, :x.shape[1]]

        # 4. Transformer 编码
        for block in self.blocks:
            x = block(x)

        # 5. 取 CLS 输出作为音频表征
        audio_repr = self.norm(x[:, 0])  # [B, D]
        return audio_repr
```

## 3. CLAP 模型（model.py）

### 3.1 双塔对比学习

```python
class CLAPModel(nn.Module):
    def forward(self, mel_spec, token_ids):
        # 音频塔
        audio_feat = self.audio_encoder(mel_spec)    # [B, D_a]
        audio_proj = self.audio_proj(audio_feat)     # [B, D_proj]
        audio_proj = F.normalize(audio_proj, dim=-1)

        # 文本塔
        text_feat = self.text_encoder(token_ids)     # [B, D_t]
        text_proj = self.text_proj(text_feat)        # [B, D_proj]
        text_proj = F.normalize(text_proj, dim=-1)

        # 对称 InfoNCE Loss
        logits = torch.matmul(audio_proj, text_proj.T) / self.temperature
        labels = torch.arange(B, device=logits.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2t + loss_t2a) / 2

        return loss, audio_proj, text_proj
```

### 3.2 全模态 Q-Former 融合

```python
class OmniModalQFormer(nn.Module):
    def forward(self, image_feats, text_feats, audio_feats):
        # 可学习 Query
        queries = self.query_tokens.expand(B, -1, -1)  # [B, M, D]

        # 拼接所有模态 KV
        modality_tokens = []
        if image_feats is not None:
            modality_tokens.append(image_feats + self.image_type_embed)
        if text_feats is not None:
            modality_tokens.append(text_feats + self.text_type_embed)
        if audio_feats is not None:
            modality_tokens.append(audio_feats + self.audio_type_embed)

        kv = torch.cat(modality_tokens, dim=1)  # [B, N_total, D]

        # Cross-Attention: Q=queries, KV=多模态
        for layer in self.cross_layers:
            queries = layer(queries, kv)

        # 池化输出
        output = queries.mean(dim=1)  # [B, D]
        return output
```

## 4. 模态缺失处理

```python
class ModalityDropout(nn.Module):
    """训练时随机丢弃模态，增强鲁棒性"""

    def forward(self, image_feats, text_feats, audio_feats, p_drop=0.1):
        if self.training:
            # 随机决定是否保留每个模态
            keep_image = random.random() > p_drop
            keep_text = random.random() > p_drop
            keep_audio = random.random() > p_drop

            # 至少保留一个模态
            if not (keep_image or keep_text or keep_audio):
                keep_text = True

            if not keep_image: image_feats = None
            if not keep_text: text_feats = None
            if not keep_audio: audio_feats = None

        return image_feats, text_feats, audio_feats
```

## 5. 音频事件检测

```python
class AudioEventDetector(nn.Module):
    """帧级 + clip 级音频事件检测"""

    def forward(self, audio_features):
        # audio_features: [B, T, D] 每帧特征

        # 帧级预测
        frame_logits = self.frame_head(audio_features)  # [B, T, C]

        # Clip 级预测（注意力池化）
        attn_weights = self.attn_pool(audio_features)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (audio_features * attn_weights).sum(dim=1)  # [B, D]
        clip_logits = self.clip_head(pooled)  # [B, C]

        return frame_logits, clip_logits
```
