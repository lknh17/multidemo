# V15 - 视频理解与 Dense Captioning

## 核心知识点
- 视频时序建模：3D Conv / TimeSformer / Video Swin
- Dense Video Captioning：时序定位 + 描述生成
- 时序提议网络（Temporal Proposal Network）
- 端到端 Dense Caption 模型：PDVC / Vid2Seq
- 时序 Grounding：Moment Retrieval / Temporal Sentence Grounding
- 优化方案：时序 NMS、Deformable 时序注意力、分层时序聚合
- 视频-语言预训练：VideoBERT / VideoMAE / InternVideo
- 长视频处理：Token Merging / 关键帧采样策略

## 运行方式

```bash
# 训练 Dense Video Captioning 模型
python train.py --mode dense_caption

# 训练时序 Grounding 模型
python train.py --mode temporal_grounding

# 训练视频编码器
python train.py --mode video_encoder

# 推理与可视化
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 视频理解与 Dense Caption 配置 |
| video_encoder.py | 视频编码器（3D Conv / TimeSformer / Video Swin） |
| model.py | Dense Caption / 时序 Grounding 端到端模型 |
| dataset.py | 视频 Dense Caption 数据集 |
| train.py | 训练脚本（多模式） |
| inference.py | 推理与时序可视化实验 |
| theory.md | 数学原理与优化方案详解 |
| code_explained.md | 代码实现详解 |
| theory_visual.html | 原理动画可视化 |
| code_visual.html | 代码动画可视化 |
