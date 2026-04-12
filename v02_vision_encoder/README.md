# v02: 视觉编码器 ViT

## 任务
用从零实现的 ViT 在 CIFAR-10 上训练图像分类

## 快速开始
```bash
pip install -r requirements.txt
python train.py       # 训练 (CIFAR-10 自动下载)
python inference.py   # 推理 + 可视化
```

## 预期输出
- 30 epoch 后验证准确率 ~80%+
- 生成预测可视化图 `logs/predictions.png`
