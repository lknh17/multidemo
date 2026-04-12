# p10 最佳实践总结

> **目标**: 汇总全流程最佳实践，提供故障排查工具和一键流水线。
>
> **前置条件**: 完成 p01-p09 全部模块。
>
> **预计耗时**: 1-2 小时（阅读 + 运行诊断工具）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 获得各阶段推荐配置的速查表，不必每次翻回前面的模块
- 掌握一套系统化的故障排查方法（GPU/Loss/环境问题）
- 能用一键脚本串联 p02→p03→p04→p05 全流程训练
- 通过终极对比直观看到全流程训练带来的效果提升

### 本模块代码文件概览

本模块是全系列的"总控台"：**配置速查 + 故障诊断 + 一键流水线 + 终极对比**。不引入新知识，而是将前面学到的所有内容整合为可操作的工具。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 各阶段推荐超参数汇总（学习率/Batch/Epoch/LoRA 配置） | 快速查阅最佳参数 |
| `troubleshooting.py` | 系统化故障诊断：GPU 状态/Loss 异常/训练卡住/依赖问题 | 掌握故障排查方法 |
| `pipeline_runner.py` | 一键串联 p02→p05 全流程，支持跳过/断点恢复/dry-run | 自动化训练流水线 |
| `inference.py` | 加载各阶段模型，用相同 prompt 并排对比效果 | 直观感受全流程成果 |

> 💡 **建议学习顺序**: 先读 `config.py` 速查表 → 跑 `troubleshooting.py` 检查环境 → `pipeline_runner.py --dry-run` 查看计划 → `inference.py` 终极对比

---

## 2. 各阶段推荐配置速查

| 阶段 | 学习率 | Batch | Epoch | LoRA r | 特殊设置 |
|------|--------|-------|-------|--------|----------|
| 继续预训练 | 2e-5 | 4×4=16 | 1-3 | 64 | Packing, 混入通用数据 |
| SFT 微调 | 2e-5 | 4×4=16 | 2-5 | 16 | NEFTune, Chat Template |
| DPO 对齐 | 5e-6 | 2×8=16 | 1-2 | 16 | beta=0.1, pair 数据 |
| RL (GRPO) | 1e-6 | 2×8=16 | 1 | - | 奖励函数, KL 惩罚 |

```bash
python -c "from config import config; print(f'SFT lr: {config.sft.learning_rate}')"
```

---

## 3. 故障排查工具

```bash
# 全面诊断
python troubleshooting.py

# 单项检查
python troubleshooting.py --check gpu
python troubleshooting.py --check loss --log-file outputs/trainer_state.json
python troubleshooting.py --check env
```

诊断工具会检测：GPU 状态、Loss 异常、训练卡住、环境依赖等问题，并给出具体修复建议。

---

## 4. 一键全流程

```bash
# 完整流水线: p02 → p03 → p04 → p05
python pipeline_runner.py --dry-run    # 先查看计划

python pipeline_runner.py              # 实际执行

# 跳过某些阶段
python pipeline_runner.py --skip p02

# 从断点恢复
python pipeline_runner.py --resume-from p04

# 重置进度
python pipeline_runner.py --reset
```

---

## 5. 终极模型对比

```bash
# 对比所有阶段
python inference.py

# 只对比特定阶段
python inference.py --stages base sft rl
```

用相同的 prompt 对比各阶段模型，直观感受全流程训练效果。
