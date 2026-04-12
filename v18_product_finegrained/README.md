# V18 - 商品理解与细粒度视觉

## 核心知识点
- 细粒度视觉识别（Fine-Grained Visual Recognition）
- 商品属性提取：类目/品牌/颜色/材质/风格
- 多粒度特征学习：局部 + 全局
- 注意力裁剪（Attention Cropping）& 零件检测
- 商品图像质量评估
- 跨域商品匹配（同款不同图）

## 运行方式

```bash
python train.py --mode fine_grained
python train.py --mode attribute
python train.py --mode quality
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 商品理解配置 |
| fine_grained.py | 细粒度识别核心模块 |
| model.py | 商品属性提取 / 质量评估模型 |
| dataset.py | 合成商品数据集 |
| train.py | 训练脚本 |
| inference.py | 推理实验 |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画 |
| code_visual.html | 代码动画 |
