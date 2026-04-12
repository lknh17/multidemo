# p09 评测体系

> **目标**: 构建完整的模型评测体系，对比五个训练阶段（基座/预训练/SFT/DPO/RL）在知识、对话、数学、代码、安全五个维度的表现。
>
> **前置条件**: 完成 p01-p05 各阶段训练，拥有五个阶段的模型 checkpoint。
>
> **预计耗时**: 2-4 小时（含全部评测和图表生成）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解主流评测基准（MMLU/C-Eval/GSM8K/HumanEval）的设计思路
- 掌握 few-shot 评测的 prompt 构建方法
- 实际运行四大评测基准并获得量化分数
- 设计自定义评测维度和评分标准
- 用雷达图直观展示五阶段模型的能力变化
- 理解评测中的常见陷阱（数据泄露、prompt 敏感性等）

### 本模块代码文件概览

本模块构建完整的评测体系：**四大标准基准 + 自定义评测 + 雷达图可视化**。每个评测脚本独立运行，最终汇总到雷达图中直观对比。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 评测全局配置（模型路径、few-shot 数、评测维度） | 理解评测参数设置 |
| `eval_mmlu.py` | MMLU 英文知识评测（57 学科 few-shot 选择题） | 掌握知识能力评测方法 |
| `eval_ceval.py` | C-Eval 中文知识评测（中国特色学科） | 评估中文知识能力 |
| `eval_gsm8k.py` | GSM8K 数学推理评测（CoT 生成 + 答案提取） | 评估数学推理能力 |
| `eval_humaneval.py` | HumanEval 代码评测（函数生成 + 测试用例验证） | 评估代码生成能力 |
| `eval_custom.py` | 自定义评测（五维度评分：准确/流畅/相关/安全/完整） | 学会设计自定义评测 |
| `radar_chart.py` | 生成五阶段模型的五维雷达图 | 直观展示训练效果 |
| `inference.py` | 多阶段模型并排对比推理 | 定性感受各阶段差异 |

> 💡 **建议学习顺序**: 先跑 `eval_mmlu.py` 和 `eval_ceval.py` 了解 few-shot 评测 → `eval_gsm8k.py` 看生成式评测 → `eval_humaneval.py` → `eval_custom.py` → `radar_chart.py` 汇总

### 确认前置条件

```bash
cd p09_evaluation
python -c "from config import config; print(f'评测维度: {config.radar.dimensions}')"
# 预期输出: 评测维度: ['知识问答', '对话能力', '数学推理', '代码生成', '安全性']
```

---

## 2. MMLU 英文知识评测

```bash
python eval_mmlu.py --model Qwen/Qwen2.5-0.5B --num-few-shot 5 --max-samples 50
```

MMLU 涵盖 57 个学科，本模块选取 9 个代表性学科进行评测。

**评测方法**: 给定 few-shot 示例 + 当前问题，比较 A/B/C/D 四个 token 的 logit。

**预期结果**:

| 阶段 | 准确率 |
|------|--------|
| 基座 | ~25-35% |
| SFT | ~35-45% |
| DPO/RL | ~40-48% |

---

## 3. C-Eval 中文知识评测

```bash
python eval_ceval.py --model Qwen/Qwen2.5-0.5B --num-few-shot 5
```

C-Eval 是中文评测基准，涵盖中国特色学科。评测方法与 MMLU 类似。

观察要点：
- 继续预训练后中文知识是否提升
- SFT 是否影响基础知识能力

---

## 4. GSM8K 数学推理评测

```bash
python eval_gsm8k.py --model Qwen/Qwen2.5-0.5B --max-samples 50
```

GSM8K 评测数学推理能力，要求模型生成 Chain-of-Thought 推理过程。

**关键**: 从生成文本中提取最终数字答案，与标准答案比较。

**预期**: RL 训练后数学能力应有显著提升。

---

## 5. HumanEval 代码评测

```bash
python eval_humaneval.py --model Qwen/Qwen2.5-0.5B --k 1 5
```

给定函数签名和 docstring，模型需生成正确的函数体。通过运行测试用例验证正确性。

**指标**: pass@k — 生成 k 个样本中至少一个通过测试的概率。

---

## 6. 自定义评测

```bash
python eval_custom.py --model Qwen/Qwen2.5-0.5B
```

支持用户自定义 prompt 和评分标准，包含：
- 知识问答、对话能力、数学推理、代码生成、安全性

每个回答从五个维度评分：准确性、流畅性、相关性、安全性、完整性。

---

## 7. 雷达图对比

```bash
python radar_chart.py --demo
```

生成五阶段模型在五个维度上的雷达图，直观展示训练各阶段的能力变化。

---

## 8. 五阶段对比推理

```bash
python inference.py --stages base sft dpo
```

加载多个阶段的模型，用相同 prompt 并排对比，直观感受训练效果。

---

## 9. 评测陷阱与注意事项

| 陷阱 | 说明 | 应对 |
|------|------|------|
| Benchmark 泄露 | 训练数据中包含评测题目 | 检查数据去重 |
| Prompt 敏感性 | 不同 prompt 格式导致结果差异大 | 多种格式取平均 |
| 过拟合评测集 | 针对特定 benchmark 优化 | 使用多个评测集 |
| 样本量不足 | 太少样本导致方差大 | 至少 100+ 样本 |

---

## 10. 小结

- [x] 运行 MMLU/C-Eval 知识评测
- [x] 运行 GSM8K 数学推理评测
- [x] 运行 HumanEval 代码评测
- [x] 设计并运行自定义评测
- [x] 生成雷达图对比五阶段模型
- [x] 理解评测中的常见陷阱

**下一步**: 进入 [p10 最佳实践总结](../p10_best_practices/README.md)，汇总全流程经验。
