# V17 - 音频理解与全模态模型

## 核心知识点
- 音频特征提取：Mel 频谱 / MFCC / Wav2Vec2
- 音频 Transformer：AST（Audio Spectrogram Transformer）
- 语音识别基础：CTC / Attention / Transducer
- 音频-文本对齐：CLAP（Contrastive Language-Audio Pretraining）
- 全模态融合：图像+文本+音频联合编码
- 音频事件检测与分类
- 音视频同步与对齐

## 运行方式

```bash
# 训练音频编码器
python train.py --mode audio_encoder

# 训练 CLAP 音频-文本对齐
python train.py --mode clap

# 训练全模态融合
python train.py --mode omni_modal

# 推理与可视化
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 音频与全模态配置 |
| audio_modules.py | 音频特征提取 + AST 编码器 |
| model.py | CLAP / 全模态融合模型 |
| dataset.py | 合成音频数据集 |
| train.py | 训练脚本（多模式） |
| inference.py | 推理与可视化实验 |
| theory.md | 数学原理详解 |
| code_explained.md | 代码实现详解 |
| theory_visual.html | 原理动画可视化 |
| code_visual.html | 代码动画可视化 |
