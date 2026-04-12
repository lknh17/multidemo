# V19 - 层级标签理解 / 广告商品标签体系

## 核心知识点
- 层级标签分类：从粗到细的多级分类（行业→品类→子品类→SKU）
- 分类学树（Taxonomy Tree）构建与遍历
- 层级 Softmax（Hierarchical Softmax）
- 条件概率链 P(fine|coarse, x) 建模
- 标签传播（Label Propagation）图神经网络
- 双曲空间标签嵌入（Poincaré Ball Model）
- 多标签分类 + 层级约束（Hierarchy-constrained Multi-label）
- 标签平滑与分类学损失（Taxonomic Loss）

## 运行方式

```bash
# 训练层级分类器（coarse→mid→fine 三级）
python train.py --mode hierarchical

# 训练多标签分类模型（层级约束）
python train.py --mode multi_label

# 训练标签嵌入模型（视觉-标签联合嵌入）
python train.py --mode label_embed

# 推理与可视化实验
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 层级标签配置（树深度、嵌入维度等） |
| taxonomy.py | 分类学树、层级 Softmax、标签传播 GNN、双曲嵌入 |
| model.py | 层级分类器 / 多标签模型 / 标签嵌入模型 |
| dataset.py | 合成层级标签数据集 |
| train.py | 训练脚本（三种模式） |
| inference.py | 推理与可视化实验 |
| theory.md | 数学原理详解 |
| code_explained.md | 代码实现详解 |
| theory_visual.html | 原理动画可视化 |
| code_visual.html | 代码动画可视化 |
