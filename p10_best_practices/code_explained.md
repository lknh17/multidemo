# p10 代码详解 - 最佳实践总结

> 逐文件解释最佳实践模块的核心代码实现。

---

## 1. config.py — 分阶段推荐配置

### 配置层级设计

```python
@dataclass
class BestPracticesConfig:
    pretrain: PretrainBestConfig = field(default_factory=PretrainBestConfig)
    sft: SFTBestConfig = field(default_factory=SFTBestConfig)
    dpo: DPOBestConfig = field(default_factory=DPOBestConfig)
    rl: RLBestConfig = field(default_factory=RLBestConfig)
    mllm: MLLMBestConfig = field(default_factory=MLLMBestConfig)
    quant: QuantBestConfig = field(default_factory=QuantBestConfig)
    deploy: DeployBestConfig = field(default_factory=DeployBestConfig)
```

**设计要点**:
- 每个阶段独立的 dataclass，修改一个阶段不影响其他
- 所有推荐值直接写在默认参数中，即开即用
- 用户可以覆盖任意参数：`config.sft.learning_rate = 1e-5`

### 学习率递减规律

```python
pretrain: 2e-5   # 最大（需要学习大量知识）
sft:      2e-5   # 中等（学习对话格式）
dpo:      5e-6   # 较小（微调偏好，避免破坏已有能力）
rl:       1e-6   # 最小（强化学习极易不稳定）
```

**核心原则**: 越后面的阶段越"精细"，学习率越小，避免灾难性遗忘。

---

## 2. troubleshooting.py — 故障诊断系统

### GPU OOM 诊断

```python
def check_gpu():
    total_mem = props.total_mem / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    usage_ratio = reserved / total_mem
    
    if usage_ratio > 0.95:
        suggestions.append("开启 gradient_checkpointing=True")
        suggestions.append("减小 batch_size 或 max_seq_length")
```

**分级预警**: 95% 以上为临界 OOM，85% 以上为高风险。根据严重程度给出不同建议。

### Loss 异常检测

```python
def check_loss(losses):
    # 1. NaN 检测
    nan_indices = [i for i, l in enumerate(losses) if math.isnan(l)]
    
    # 2. 不下降检测
    first_quarter_avg vs last_quarter_avg
    
    # 3. 震荡检测
    volatility = avg_diff / avg_loss
    
    # 4. 突然飙升检测
    if losses[i] > losses[i-1] * 2
```

**四级诊断**: NaN（最严重）→ 飙升 → 不下降 → 震荡，每种异常给出不同的修复建议。

### 环境检测

```python
def check_environment():
    # Python / PyTorch / Transformers / Flash-Attn 版本
    # 磁盘空间
    # CUDA 可用性
```

**全面覆盖**: 一个命令检测所有常见环境问题，避免训练到一半才发现环境配置不对。

---

## 3. pipeline_runner.py — 流水线调度

### 断点恢复机制

```python
PROGRESS_FILE = "outputs/pipeline_progress.json"

def load_progress():
    return {"completed": ["p02", "p03"], "current": "p04"}

def run_pipeline(resume_from=None):
    for stage in PIPELINE_STAGES:
        if stage["id"] in progress["completed"]:
            continue  # 跳过已完成的
        
        success = run_stage(stage)
        if success:
            progress["completed"].append(stage["id"])
            save_progress(progress)
        else:
            print(f"修复后运行: --resume-from {stage['id']}")
            return  # 失败时停止
```

**设计要点**:
1. **进度持久化**: 用 JSON 文件记录已完成的阶段
2. **自动恢复**: 重新运行时自动跳过已完成阶段
3. **手动恢复**: `--resume-from` 从指定阶段开始
4. **阶段跳过**: `--skip p02` 跳过不需要的阶段

### 子进程执行与超时

```python
result = subprocess.run(
    [sys.executable, script_path],
    cwd=os.path.dirname(script_path),
    timeout=7200,  # 2 小时超时
)
```

每个阶段在子进程中执行，好处：
- 阶段间内存完全隔离
- 一个阶段崩溃不影响流水线进程
- 可以设置超时限制
