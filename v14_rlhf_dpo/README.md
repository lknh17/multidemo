# v14 - RLHF / DPO 偏好对齐

## 核心知识点
1. **Reward Model** — Bradley-Terry 偏好模型，从人类比较数据训练奖励函数
2. **PPO (RLHF)** — 近端策略优化，用 Reward Model 指导 LLM 策略更新
3. **DPO** — 直接偏好优化，跳过 Reward Model，直接从偏好数据优化策略
4. **SimPO / ORPO / KTO** — 最新偏好对齐变体，无需参考模型
5. **广告偏好数据构造** — 如何为广告多模态理解构造偏好对
6. **Embedding 偏好对齐** — 将 DPO 思想应用到 Embedding 空间

## 快速开始
```bash
# 训练 Reward Model
python train.py --mode reward

# DPO 训练
python train.py --mode dpo

# 推理对比
python inference.py
```

## 文件结构
- `config.py` — 偏好对齐超参数配置
- `model.py` — Reward Model + DPO 策略模型
- `losses.py` — DPO/SimPO/KTO 各种 Loss 实现
- `dataset.py` — 偏好对数据集
- `train.py` — 多模式训练脚本
- `inference.py` — 对齐前后对比推理
- `theory.md` — RLHF→DPO 完整数学推导
- `code_explained.md` — 代码实现详解
- `theory_visual.html` — 原理动画可视化
- `code_visual.html` — 代码动画可视化
