# V24 - 内容安全与合规

## 核心知识点
- 内容安全分类（NSFW/暴力/垃圾/仇恨 等 8 类）
- 多模态毒性检测（文本 + 图像融合评分）
- 水印嵌入与检测（DWT/DCT 域嵌入）
- 对抗鲁棒性（FGSM / PGD 攻击与对抗训练）
- LLM 输出安全护栏（Guardrails）
- 合规检查（规则 + 模型级联）
- 校准与可靠性（Platt Scaling / 温度缩放）

## 运行方式

```bash
python train.py --mode safety_cls
python train.py --mode adversarial
python train.py --mode watermark
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 安全检测配置 |
| safety_modules.py | 内容分类 / 毒性评分 / 水印 / 对抗攻击核心模块 |
| model.py | SafetyGuard 级联模型 / 合规检查 / 校准分类器 |
| dataset.py | 合成安全标注数据集 |
| train.py | 训练脚本 |
| inference.py | 推理实验 |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画 |
| code_visual.html | 代码动画 |
