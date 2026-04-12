# V22 - 评估体系与 A/B 测试

## 核心知识点
- 离线评估指标（Recall@K、NDCG、MRR、mAP）
- 在线 A/B 测试框架与流量分割
- 统计显著性检验（Welch's t-test、Bootstrap CI）
- 多臂老虎机（Epsilon-Greedy、UCB1、Thompson Sampling）
- 交错实验（Team Draft Interleaving）
- 公平性指标（Demographic Parity、Equalized Odds）

## 运行方式

```bash
python train.py --mode offline_eval
python train.py --mode ab_test
python train.py --mode bandit
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 评估与 A/B 测试配置 |
| metrics.py | 检索 / 分类 / 公平性评估指标 |
| model.py | A/B 测试框架 / Bandit 选择器 / 交错实验 |
| dataset.py | 合成实验日志与指标数据 |
| train.py | 训练脚本（离线评估 / A/B 测试 / Bandit） |
| inference.py | 推理实验 |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画 |
| code_visual.html | 代码动画 |
