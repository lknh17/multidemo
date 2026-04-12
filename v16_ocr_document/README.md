# V16 - OCR 与文档理解 / 广告文字提取

## 核心知识点
- OCR 基础：检测（EAST/DBNet）+ 识别（CRNN/ABINet）
- 文档理解：LayoutLM / LayoutLMv3 / DocFormer
- 2D 位置编码：空间布局感知
- 多模态文档预训练：MLM + MIM + WPA
- 广告文字提取：Logo OCR、促销文案识别、价格标签解析
- 端到端场景文字识别（Scene Text Recognition）
- 表格结构识别（Table Structure Recognition）

## 运行方式

```bash
# 训练 OCR 文字检测+识别模型
python train.py --mode ocr

# 训练文档理解模型
python train.py --mode document

# 训练广告文字提取模型
python train.py --mode ad_text

# 推理与可视化
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | OCR 与文档理解配置 |
| ocr_modules.py | OCR 检测 + 识别核心模块 |
| model.py | 文档理解 / 广告文字端到端模型 |
| dataset.py | 合成 OCR / 文档数据集 |
| train.py | 训练脚本（多模式） |
| inference.py | 推理与可视化实验 |
| theory.md | 数学原理详解 |
| code_explained.md | 代码实现详解 |
| theory_visual.html | 原理动画可视化 |
| code_visual.html | 代码动画可视化 |
