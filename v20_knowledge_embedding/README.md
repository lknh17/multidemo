# V20 - 知识增强多模态嵌入（Knowledge-Enhanced Multimodal Embedding）

## 核心知识点
- 知识图谱嵌入：TransE / TransR 表示学习
- 实体链接（Entity Linking）：mention 检测 + 候选排序
- 知识增强注意力：将实体嵌入注入为额外 KV 对
- KG 增强视觉特征：图神经网络（GNN）消息传递
- 关系推理（Relation Reasoning）：多跳推理路径
- 知识蒸馏：从 KG 到视觉模型的知识迁移

## 运行方式

```bash
# 训练知识图谱嵌入（TransE / TransR）
python train.py --mode kg_embed

# 训练 KG 增强视觉模型
python train.py --mode kg_visual

# 训练知识蒸馏模型
python train.py --mode distill

# 推理与可视化实验
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 知识图谱与 KG 增强嵌入配置 |
| kg_modules.py | KG 嵌入 / GNN / 实体链接 / KG 注意力核心模块 |
| model.py | KG 增强视觉模型 / KG 增强检索 / 知识蒸馏模型 |
| dataset.py | 合成 KG 三元组 + 图像-实体对数据集 |
| train.py | 训练脚本（三种模式） |
| inference.py | 推理与可视化实验 |
| theory.md | 数学原理详解 |
| code_explained.md | 代码实现详解 |
| theory_visual.html | 原理动画可视化 |
| code_visual.html | 代码动画可视化 |
