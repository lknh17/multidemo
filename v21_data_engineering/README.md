# V21 - 多模态数据工程

## 核心知识点
- 数据质量过滤（基于 CLIP 质量评分、分辨率、宽高比）
- 数据去重：MinHash / SimHash
- 数据增强：CutMix / MixUp / RandAugment
- 课程学习（Curriculum Learning）与难度调度
- 数据平衡：类别权重 / 过采样策略
- 合成数据生成（简单条件生成）

## 运行方式

```bash
pip install -r requirements.txt
python train.py --mode augmented
python train.py --mode curriculum
python train.py --mode balanced
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 数据工程全流程配置 |
| data_ops.py | 去重（MinHash/SimHash）/ 质量评分 / 数据平衡 |
| model.py | 增强训练器 / 课程调度器 / 合成数据生成器 |
| dataset.py | 合成数据集（演示管道各阶段） |
| train.py | 三种训练模式：augmented / curriculum / balanced |
| inference.py | 四组实验：去重压缩率 / 增强效果 / 课程 vs 随机 / 质量分布 |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画（6步） |
| code_visual.html | 代码动画（5步） |

## 预期输出
- 去重后数据量压缩 30-50%
- 数据增强使模型精度提升 3-8%
- 课程学习比随机训练收敛更快
- 质量评分呈近似正态分布
