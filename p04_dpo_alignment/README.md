# p04 DPO 偏好对齐训练

> **目标**: 使用 DPO/SimPO/ORPO/KTO 四种直接偏好优化算法，对 SFT 模型进行人类偏好对齐，对比各算法在安全性、指令遵循、生成质量上的差异。
>
> **前置条件**: 完成 p03 SFT 微调，或使用 Qwen2.5-0.5B-Instruct 作为起点。
>
> **预计耗时**: 3-5 小时（含四种算法对比实验）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解 RLHF → DPO 的演进路径和动机
- 掌握 DPO、SimPO、ORPO、KTO 四种对齐算法的核心差异
- 实际运行四种算法的对比训练实验
- 理解 beta 超参数对对齐效果的影响（消融实验）
- 掌握偏好数据的获取和构造方法
- 对比 SFT vs 对齐模型在安全性和质量上的差异
- 理解 LoRA + DPO 的高效训练范式

### 本模块代码文件概览

本模块围绕偏好对齐展开：**获取偏好数据 → 四种算法训练 → Beta 消融 → 推理对比**。你将亲手运行 DPO/SimPO/ORPO/KTO 四种算法，直观比较它们在安全性和质量上的差异。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 对齐训练配置，含 DPO/SimPO/ORPO/KTO 各算法参数 | 理解四种算法的核心超参数 |
| `download_data.py` | 下载 UltraFeedback / HH-RLHF 偏好数据集 | 获取配对偏好数据 |
| `dataset.py` | 将偏好数据转为 DPOTrainer 需要的 prompt/chosen/rejected 格式 | 掌握偏好数据处理 |
| `build_preference.py` | 用 SFT 模型做 Reject Sampling，自动构造偏好对 | 学会从零构造偏好数据 |
| `train.py` | 主训练脚本，通过 `--algorithm` 切换四种对齐算法 | 运行四种算法的对比实验 |
| `inference.py` | 对比 SFT 基线与各对齐算法在安全性、质量上的差异 | 直观看到对齐效果 |

> 💡 **建议学习顺序**: `download_data.py` → `train.py --algorithm dpo`（先跑通）→ 依次尝试 simpo/orpo/kto → Beta 消融 → `build_preference.py`（可选）→ `inference.py`

### 确认前置条件

```bash
cd p04_dpo_alignment
python -c "from trl import DPOTrainer; print('trl 版本:', __import__('trl').__version__)"
# 预期输出: trl 版本: 0.12.x

python -c "from peft import LoraConfig; print('peft OK')"
# 预期输出: peft OK
```

---

## 2. 偏好数据下载与探索

```bash
# 下载 UltraFeedback 偏好数据
python download_data.py --dataset ultrafeedback --max-samples 10000

# 下载 HH-RLHF 数据
python download_data.py --dataset hh_rlhf --max-samples 5000

# 两个数据集都下载
python download_data.py --dataset both --max-samples 5000

# 查看已下载数据统计
python download_data.py --explore-only
```

**预期输出**:
```
  总数据对:        10000
  Prompt 平均长度:  186 字符
  Chosen 平均长度:  423 字符
  Rejected 平均长度:312 字符
```

### UltraFeedback vs HH-RLHF 对比

| 数据集 | 来源 | 偏好标注 | 语言 | 特点 |
|--------|------|----------|------|------|
| UltraFeedback | GPT-4 评分 | 多维度打分 | 英文 | 质量高、维度丰富 |
| HH-RLHF | 人类标注 | 二选一 | 英文 | Anthropic 经典数据 |

---

## 3. 从 SFT 模型构造偏好数据

如果你有自己的 SFT 模型，可以通过 Reject Sampling 自动构造偏好数据：

```bash
# 使用 SFT 模型生成多个回复，评分后构造偏好对
python build_preference.py \
    --model-path ../p03_sft_finetuning/outputs/sft/final \
    --num-responses 4 \
    --scoring rules \
    --pair-strategy best_worst \
    --output data/preference_built.jsonl
```

**Reject Sampling 流程**:

1. 对每个 prompt 用高 temperature 采样生成 4 个不同回复
2. 使用评分函数对回复质量打分
3. 取得分最高和最低的回复作为 chosen/rejected 对
4. 保存为标准偏好数据格式

---

## 4. 实验一：DPO 标准训练

```bash
python train.py --algorithm dpo
```

**预期输出**:
```
  训练配置:
    算法:           DPO
    Beta:           0.1
    数据集大小:     9500 (验证: 500)
    Batch Size:     2 × 8
    学习率:         5e-07
    LoRA:           是
```

观察要点：
- DPO loss（`rewards/chosen` 和 `rewards/rejected`）应逐步分离
- `rewards/margins` 应逐步增大（chosen 与 rejected 的差距加大）
- 如果 loss 震荡或 NaN，降低学习率或增大 beta

---

## 5. 实验二：SimPO（无参考模型）

```bash
python train.py --algorithm simpo
```

SimPO vs DPO 关键差异：

| 特性 | DPO | SimPO |
|------|-----|-------|
| 参考模型 | 需要（内存 ×2） | 不需要 |
| Log-prob 计算 | Token-level | 序列平均 |
| Margin | 无 | γ 参数控制 |
| 显存占用 | 较高 | 较低 |
| 论文 | Rafailov 2023 | Meng 2024 |

---

## 6. 实验三：ORPO（统一目标）

```bash
python train.py --algorithm orpo
```

ORPO 的独特之处：
- 不需要参考模型
- 不需要单独的 SFT 阶段（将 NLL + 偏好优化合为一个 loss）
- 使用 Odds Ratio 替代 log probability ratio
- 适合从零开始训练对齐模型

---

## 7. 实验四：KTO（无需配对数据）

```bash
python train.py --algorithm kto
```

KTO 的优势：
- 不需要 chosen/rejected 配对
- 只需标注每个回复是"好"还是"坏"
- 基于 Kahneman-Tversky 前景理论
- 数据收集成本更低

---

## 8. 实验五：Beta 消融实验

```bash
# 小 beta = 弱约束（更容易偏离参考模型）
python train.py --algorithm dpo --beta 0.01

# 标准 beta
python train.py --algorithm dpo --beta 0.1

# 大 beta = 强约束（更保守，接近参考模型）
python train.py --algorithm dpo --beta 0.5
```

**Beta 消融记录**:

| Beta | 收敛速度 | 对齐效果 | 生成多样性 | 安全性 |
|------|----------|----------|------------|--------|
| 0.01 | 快 | 强但可能过拟合 | 低 | 最高 |
| 0.05 | 较快 | 较强 | 中 | 高 |
| 0.1 | 中 | 适中 | 中 | 中 |
| 0.2 | 较慢 | 较弱 | 较高 | 中 |
| 0.5 | 慢 | 弱 | 高 | 较低 |

> 💡 **结论**: beta 控制"偏离参考模型的代价"。过小会过拟合偏好数据，过大则对齐效果不明显。推荐从 0.1 开始。

---

## 9. 四种算法对比推理

```bash
# 对比 DPO vs SFT 基线
python inference.py --models dpo

# 对比所有算法
python inference.py --models dpo simpo orpo kto

# 指定模型路径
python inference.py --dpo-path outputs/dpo/dpo/final --simpo-path outputs/dpo/simpo/final
```

**对比维度**:

| 维度 | SFT 基线 | DPO | SimPO | ORPO | KTO |
|------|----------|-----|-------|------|-----|
| 安全性 | 低 | 高 | 高 | 中 | 高 |
| 指令遵循 | 中 | 高 | 高 | 高 | 中 |
| 生成质量 | 中 | 高 | 高 | 高 | 中 |
| 训练效率 | — | 中 | 高 | 高 | 中 |
| 数据需求 | — | 配对 | 配对 | 配对 | 单条 |

---

## 10. 算法选型指南与总结

### 选哪个算法？

```
需要配对数据？
├── 是 → 有参考模型预算？
│       ├── 是 → DPO（最经典、最稳定）
│       └── 否 → SimPO（省显存）或 ORPO（合并 SFT）
└── 否 → KTO（只需好/坏标签）
```

### 训练建议

1. **数据质量 > 算法选择**: 高质量偏好数据比选对算法更重要
2. **从 DPO + LoRA 开始**: 最经典的组合，调参经验最丰富
3. **Beta 调优是关键**: 对效果影响巨大，建议做消融实验
4. **小数据也能有效**: 5000-10000 条高质量偏好数据就能显著改善
5. **评估要全面**: 不只看生成质量，还要测安全性、指令遵循

### 实验记录表

| 实验 | 算法 | Beta | 数据量 | Loss | 安全性 | 质量 |
|------|------|------|--------|------|--------|------|
| #1 | DPO | 0.1 | 10K | | | |
| #2 | SimPO | 2.0 | 10K | | | |
| #3 | ORPO | 1.0 | 10K | | | |
| #4 | KTO | 0.1 | 10K | | | |
| #5 | DPO | 0.01 | 10K | | | |
| #6 | DPO | 0.5 | 10K | | | |

---

## 小结

- [x] 下载并探索了 UltraFeedback / HH-RLHF 偏好数据
- [x] 学会了通过 Reject Sampling 构造偏好数据
- [x] 运行了 DPO / SimPO / ORPO / KTO 四种对齐算法
- [x] 完成了 Beta 消融实验，理解了正则化强度的影响
- [x] 对比了 SFT 基线与各对齐算法的效果差异

**下一步**: 进入 [p05 RL GRPO 训练](../p05_rl_grpo/README.md)，探索基于强化学习的对齐方法。
