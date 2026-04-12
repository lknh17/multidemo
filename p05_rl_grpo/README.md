# p05 强化学习 GRPO

> **目标**: 使用 GRPO (Group Relative Policy Optimization) 在 GSM8K 数学推理数据集上训练模型，对比 SFT/DPO/RL 三种方案的效果差异。
>
> **前置条件**: 完成 p03 SFT 微调和 p04 DPO 训练。
>
> **预计耗时**: 3-6 小时（含多组消融实验）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解 GRPO 的核心原理：用组内相对排名代替 critic 模型
- 设计数学推理的奖励函数（正确性 + 格式 + 长度）
- 掌握 GSM8K 数据集的处理和 Chain-of-Thought prompt 构建
- 使用 trl GRPOTrainer 完成训练
- 使用 OpenRLHF 框架作为替代方案
- 监控训练过程，检测和防止奖励欺骗
- 对比 SFT / DPO / GRPO 在数学推理上的效果

### 本模块代码文件概览

本模块是强化学习实战：**设计奖励函数 → GRPO 训练 → 消融实验 → 训练监控 → 三阶段对比**。核心在于理解"奖励驱动的学习"与 SFT/DPO 的本质区别。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | GRPO 训练配置（group_size、kl_coef、clip_ratio 等） | 理解 RL 训练核心超参数 |
| `download_data.py` | 下载 GSM8K 数学推理数据集 | 获取可验证正确性的训练数据 |
| `dataset.py` | 构建 Chain-of-Thought prompt 格式 | 掌握 RL 数据的 prompt 设计 |
| `reward.py` | 三种奖励函数实现：正确性/格式/组合奖励 | 学会设计奖励函数（RL 核心） |
| `train.py` | 使用 trl GRPOTrainer 的主训练脚本 | 运行 GRPO 训练实验 |
| `train_openrlhf.py` | OpenRLHF 框架的替代训练方案 | 了解不同 RL 框架的差异 |
| `monitor.py` | 绘制训练曲线、检测奖励欺骗、质量抽样检查 | 掌握 RL 训练监控方法 |
| `inference.py` | SFT vs DPO vs GRPO 三阶段效果对比 | 直观对比三种方案 |

> 💡 **建议学习顺序**: `download_data.py` → 先读懂 `reward.py` → `train.py`（基础实验）→ 修改参数做消融 → `monitor.py` → `inference.py`

### 确认前置条件

```bash
cd p05_rl_grpo
python -c "from trl import GRPOTrainer; print('trl 已安装')"
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',trust_remote_code=True); print(f'Tokenizer 词表: {t.vocab_size}')"
```

---

## 2. 数据下载与探索

```bash
python download_data.py --max-samples 5000
```

下载 GSM8K 数学推理数据集（约 7500 条训练、1300 条测试）。

```bash
# 查看数据统计
python download_data.py --explore-only
```

**预期输出**:
```
  训练集统计:
  总条数:       7473
  问题平均长度: 240 字符
  答案平均长度: 320 字符
  平均推理步骤: 3.8 步
```

### 为什么选 GSM8K

| 数据集 | 难度 | 规模 | 特点 | 验证方式 |
|--------|------|------|------|---------|
| GSM8K | ★★★ | 7.5K | 小学数学 | 精确数值匹配 |
| MATH | ★★★★★ | 12.5K | 竞赛数学 | LaTeX 匹配 |
| ARC | ★★ | 7.8K | 选择题 | 选项匹配 |

GSM8K 难度适中，答案可精确验证，最适合 RL 奖励设计。

---

## 3. 奖励函数设计

```python
from reward import correctness_reward, format_reward, composite_reward

# 正确性奖励
r = correctness_reward("步骤1：15-3=12\n步骤2：12+7=19\n#### 19", ground_truth=19)
# → 1.0（答案正确）

# 格式奖励
r = format_reward("步骤 1：...\n步骤 2：...\n#### 19")
# → 0.6（有步骤 + 有答案标记）

# 组合奖励
r = composite_reward(response, ground_truth=19)
# → {"total": 0.78, "correctness": 1.0, "format": 0.6, "length": 0.0}
```

三种奖励的权重配比：

| 奖励 | 权重 | 含义 |
|------|------|------|
| 正确性 | 0.6 | 答案是否正确 |
| 格式 | 0.3 | 推理步骤是否清晰 |
| 长度 | 0.1 | 防止过长响应 |

---

## 4. GRPO 训练配置

查看 `config.py` 核心参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `group_size` | 8 | 每个 prompt 采样的响应数 G |
| `temperature` | 0.7 | 采样温度 |
| `clip_ratio` | 0.2 | PPO clip 范围 ε |
| `kl_coef` | 0.05 | KL 散度惩罚系数 β |
| `learning_rate` | 5e-7 | RL 训练学习率（非常小） |

---

## 5. 实验一：基础 GRPO 训练

```bash
python train.py
```

**预期输出**:
```
  训练配置:
    数据集大小:     5000
    Group Size:     8
    Batch Size:     2 × 8 = 16
    学习率:         5e-07
    KL 系数:        0.05
    奖励函数:       composite
```

观察要点：
- 平均奖励应在前 50 步上升
- KL 散度应保持在 0.5 以下
- 如果奖励不上升，尝试增大学习率

---

## 6. 实验二：Group Size 消融

```bash
python train.py --group-size 4
python train.py --group-size 8
python train.py --group-size 16
```

**对比记录**:

| Group Size | GPU 显存 | 训练速度 | 最终奖励 | 方差 |
|------------|---------|---------|---------|------|
| G=4 | ~8 GB | 快 | 较低 | 较大 |
| G=8 | ~12 GB | 中 | 基准 | 中 |
| G=16 | ~20 GB | 慢 | 较高 | 较小 |

> 结论: G=8 是显存和效果的平衡点。G 越大，优势估计越准确，但显存消耗也更大。

---

## 7. 实验三：KL 系数消融

```bash
python train.py --kl-coef 0.01
python train.py --kl-coef 0.05
python train.py --kl-coef 0.1
```

| KL 系数 | 奖励上升速度 | 最终奖励 | 奖励欺骗风险 |
|---------|------------|---------|-------------|
| β=0.01 | 快 | 高 | 高 |
| β=0.05 | 中 | 中 | 低 |
| β=0.10 | 慢 | 低 | 很低 |

> 结论: β=0.05 平衡了训练效率和策略稳定性。

---

## 8. 实验四：奖励函数对比

```bash
python train.py --reward-type correctness
python train.py --reward-type format
python train.py --reward-type composite
```

| 奖励函数 | 准确率提升 | 格式规范 | 奖励欺骗风险 |
|---------|----------|---------|------------|
| 纯正确性 | 最高 | 差 | 高（答案捷径） |
| 纯格式 | 低 | 最好 | 中（形式化） |
| 组合奖励 | 中 | 好 | 低 |

---

## 9. OpenRLHF 替代方案

```bash
python train_openrlhf.py --reward-type composite
```

trl vs OpenRLHF 对比：

| 特性 | trl | OpenRLHF |
|------|-----|----------|
| 易用性 | ★★★★★ | ★★★ |
| 多 GPU 支持 | 基础 | 强（Ray） |
| vLLM 采样加速 | 不支持 | 支持 |
| 自定义奖励 | 简单 | 灵活 |
| 社区生态 | HuggingFace | 独立 |

---

## 10. 训练监控与奖励欺骗检测

```bash
# 绘制训练曲线
python monitor.py --log-dir outputs/grpo

# 多组实验对比
python monitor.py --compare outputs/grpo_g4 outputs/grpo_g8 outputs/grpo_g16

# 奖励欺骗检测
python monitor.py --check-hacking outputs/grpo

# 生成质量抽样检查
python monitor.py --quality-check outputs/grpo/final
```

奖励欺骗的典型信号：
1. 训练奖励持续上升，但测试准确率停滞
2. KL 散度持续增大
3. 生成文本变得不自然（如重复固定模式）

---

## 11. SFT / DPO / GRPO 效果对比

```bash
python inference.py --compare-all \
    --sft-model ../p03_sft_finetuning/outputs/sft/final \
    --dpo-model ../p04_dpo/outputs/dpo/final
```

**预期对比结果**:

| 模型 | GSM8K 准确率 | 推理质量 | 格式规范 |
|------|-------------|---------|---------|
| Base | ~5% | 差 | 无 |
| SFT | ~15% | 中 | 中 |
| DPO | ~20% | 好 | 好 |
| GRPO | ~25% | 好 | 好 |

> 结论: RL (GRPO) 在可验证任务（数学推理）上的提升最明显。

---

## 12. 小结

- [x] 下载并探索了 GSM8K 数学推理数据集
- [x] 设计了三种奖励函数（正确性/格式/组合）
- [x] 使用 trl GRPOTrainer 完成 GRPO 训练
- [x] 尝试了 OpenRLHF 替代方案
- [x] 完成了 Group Size、KL 系数、奖励函数消融实验
- [x] 监控训练过程并检测奖励欺骗
- [x] 对比了 SFT / DPO / GRPO 效果

**下一步**: GRPO 是当前最先进的 RL 方法之一（DeepSeek-R1 即使用 GRPO）。继续探索更复杂的推理任务和多模态场景。
