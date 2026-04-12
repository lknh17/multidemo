# p01 环境搭建与显存估算

> **目标**: 在云服务器上从零搭建大模型训练环境，理解 GPU 显存构成和估算方法，建立原始模型的能力基线。
>
> **前置条件**: 一台配有 NVIDIA GPU (24G+) 的云服务器，有 Linux 基础操作经验。
>
> **预计耗时**: 1-2 小时（含模型下载时间）

---

## 1. 本模块目标与前置条件

### 你将收获什么

完成本模块后，你将：

- 拥有一台完整配置好的大模型训练环境
- 理解 GPU 显存的四大组成部分及估算公式
- 掌握 DeepSpeed ZeRO Stage 0-3 的区别和选型方法
- 了解不同精度（fp32/bf16/fp16/int8/int4）的显存-速度 tradeoff
- 记录 Qwen2.5-0.5B 原始模型的各项能力基线

### 本模块代码文件概览

本模块附带以下代码文件，它们构成了一条完整的学习路径：**搭建环境 → 下载模型 → 验证环境 → 测试性能 → 估算显存 → 建立基线**。你不需要从零写代码，而是通过运行和阅读这些脚本来掌握核心概念。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `setup.sh` | 一键安装全部依赖（PyTorch、Transformers、DeepSpeed 等） | 快速搭建训练环境 |
| `config.py` | 全局配置中心，自动检测 GPU 型号并匹配训练参数 | 理解训练配置管理方式 |
| `download_model.py` | 下载 Qwen2.5-0.5B 模型（支持镜像加速和断点续传） | 获取基座模型 |
| `verify_env.py` | 逐项检测 GPU/CUDA/Python/库版本，输出详细报告 | 确认环境正确 |
| `gpu_benchmark.py` | 测试 GPU 的 TFLOPS、显存带宽、Flash Attention 加速比 | 了解硬件实际性能 |
| `vram_estimator.py` | 实现显存四大组成（参数/梯度/优化器/激活值）的估算公式 | 掌握显存估算方法 |
| `baseline_eval.py` | 用固定 prompt 在 5 个维度评测模型能力，保存 JSON 结果 | 建立可对比的能力基线 |
| `inference.py` | 交互式推理，支持 base/chat 模式对比 | 直观感受模型表现 |

> 💡 **建议学习顺序**: 先跑 `setup.sh` → `verify_env.py` → `download_model.py` → `gpu_benchmark.py` → `vram_estimator.py` → `baseline_eval.py` → `inference.py`，按照教程一步步操作。

### 前置条件

| 条件 | 要求 | 说明 |
|------|------|------|
| GPU | NVIDIA 显存 >= 24GB | 4090 / A6000 / A100 均可 |
| 系统 | Ubuntu 20.04+ | CentOS 7+ 也行，但 Ubuntu 生态更好 |
| CUDA | >= 11.8 | 运行 `nvidia-smi` 确认 |
| Python | >= 3.10 | 推荐使用 conda 管理 |
| 磁盘 | >= 50GB 可用空间 | 模型下载 + 训练输出 |

> 💡 **提示**: 如果你还没有云服务器，请先阅读第 2 节的平台选择指南。

---

## 2. 云服务器选择与租用指南

### 国内主流 GPU 云平台对比

| 平台 | 4090 (24G) 价格 | A6000 (48G) 价格 | 优点 | 缺点 |
|------|-----------------|------------------|------|------|
| **AutoDL** | ~2-3 元/时 | ~4-6 元/时 | 价格最低，镜像丰富，学术用户友好 | 高峰时段排队 |
| **恒源云 (GPUHUB)** | ~3-4 元/时 | ~5-7 元/时 | 稳定性好，客服响应快 | 价格略高 |
| **矩池云** | ~3-4 元/时 | ~5-7 元/时 | 支持 Jupyter，界面友好 | 机型选择较少 |
| **Lambda Cloud** | ~$1.1/时 (A6000) | - | 海外平台，无需科学上网 | 需美元支付，延迟高 |
| **Vast.ai** | 价格浮动 | - | 社区 GPU，价格最灵活 | 稳定性不保证 |

### 推荐选择

对于本教程系列，**推荐 AutoDL + 4090 (24G)**，原因：

1. **性价比最高**: 4090 的 fp16 算力（165 TFLOPS）远超 A6000（77 TFLOPS）
2. **教程兼容**: 本系列所有实验都针对 24G 显存优化过
3. **国内加速**: AutoDL 自带学术镜像加速，模型下载飞快

### 租用步骤（以 AutoDL 为例）

```bash
# 1. 注册账号: https://www.autodl.com
# 2. 充值（支持微信/支付宝）
# 3. 创建实例:
#    - GPU: RTX 4090 (24G)
#    - 镜像: PyTorch 2.1 + CUDA 11.8
#    - 系统盘: 30GB (默认)
#    - 数据盘: 50GB+
# 4. 启动实例后通过 SSH 或 JupyterLab 连接
```

> ⚠️ **注意**: 不用时一定要**关机**！按小时计费的云服务器，忘记关机一天可能花费 50-100 元。

---

## 3. 系统环境检查

连接到云服务器后，先确认基础环境：

```bash
# 检查 GPU 和 CUDA Driver
nvidia-smi

# 预期输出中确认:
# - GPU 型号 (如 NVIDIA GeForce RTX 4090)
# - Driver 版本 (如 535.xx)
# - CUDA 版本 (如 12.2, 这是 Driver 支持的最高 CUDA 版本)

# 检查 Python 版本
python3 --version   # 需要 3.10+

# 检查 CUDA toolkit 版本（如果预装了）
nvcc --version

# 检查磁盘空间
df -h
```

**预期输出示例**:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01   Driver Version: 535.183.01   CUDA Version: 12.2                 |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0  Off |                  Off |
|-----------------------------------------+------------------------+----------------------+
```

> 💡 **提示**: `nvidia-smi` 显示的 CUDA Version 是 Driver 支持的最高版本，不是实际安装的 CUDA Toolkit 版本。PyTorch 自带 CUDA runtime，所以通常不需要单独安装 CUDA Toolkit。

---

## 4. 一键环境安装

### 4.1 创建虚拟环境（推荐）

```bash
# 如果使用 conda
conda create -n llm python=3.10 -y
conda activate llm

# 如果使用 venv
python3 -m venv llm_env
source llm_env/bin/activate
```

### 4.2 克隆项目并安装依赖

```bash
# 克隆项目
git clone <本项目地址>
cd multidemo/p01_env_setup

# 方式一: 使用一键安装脚本（推荐）
chmod +x setup.sh
bash setup.sh

# 方式二: 手动安装核心依赖
pip install -r requirements.txt
```

### 4.3 各依赖包的作用

| 包名 | 作用 | 使用阶段 |
|------|------|---------|
| `torch` | 深度学习基础框架 | 全部 |
| `transformers` | 模型加载/训练 | 全部 |
| `datasets` | 数据集加载 | p02-p06 |
| `accelerate` | 分布式训练封装 | p02-p06 |
| `peft` | LoRA/QLoRA/DoRA | p03 |
| `trl` | DPO/GRPO 训练 | p04-p05 |
| `deepspeed` | ZeRO 显存优化 | p02-p03 |
| `bitsandbytes` | 4/8-bit 量化 | p03, p07 |
| `flash-attn` | 高效注意力（可选） | 全部（加速） |

> ⚠️ **注意**: Flash Attention 安装需要编译 CUDA 内核，可能需要 10-15 分钟。如果安装失败，训练仍可正常运行，只是速度稍慢。

---

## 5. 模型下载

### 5.1 下载基座模型

```bash
# 使用国内镜像加速（AutoDL 等国内平台推荐）
python download_model.py --model base

# 海外服务器（不需要镜像）
python download_model.py --model base --no-mirror

# 下载全部模型（base + chat + vl）
python download_model.py --model all
```

### 5.2 本教程使用的模型

| 模型 | 参数量 | 用途 | 显存占用(bf16) |
|------|--------|------|---------------|
| `Qwen/Qwen2.5-0.5B` | 0.49B | p02-p05 基座 | ~1 GB |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 效果对比 | ~1 GB |
| `Qwen/Qwen2.5-VL-2B-Instruct` | 2.2B | p06 MLLM | ~4.5 GB |

### 5.3 为什么选 Qwen2.5-0.5B

1. **参数量合适**: 0.5B 在 24G 显存下可以全参预训练 + 全参 SFT
2. **训练速度快**: 单卡 1-2 小时就能看到训练效果
3. **生态完善**: Qwen2.5 系列文档丰富，社区活跃
4. **与教学系列一脉相承**: 现有 v01-v25 以 Qwen-VL 为主线
5. **可扩展**: 学会了 0.5B，方法直接迁移到 1.5B/7B

> 💡 **提示**: 模型下载到 `~/.cache/huggingface/hub`，后续 `from_pretrained` 会自动从缓存加载，不需要重复下载。

---

## 6. 环境验证

```bash
python verify_env.py
```

**预期输出**:

```
==============================================================
  实践大模型 - 环境验证报告
==============================================================

📋 系统环境
----------------------------------------
  ✅ Python 版本: 3.10.12
  ✅ PyTorch + CUDA: PyTorch 2.4.0 | CUDA 11.8
  ✅ GPU 硬件: NVIDIA GeForce RTX 4090 x1 | 显存: 24.0GB | 架构: Ada Lovelace | bf16: 支持

📦 核心依赖
----------------------------------------
  ✅ transformers: 4.45.0
  ✅ datasets: 2.20.0
  ✅ accelerate: 0.34.0
  ✅ peft: 0.12.0
  ✅ trl: 0.11.0

🔧 训练框架
----------------------------------------
  ✅ DeepSpeed: 0.15.0 | CUDA 扩展可用
  ✅ Flash Attention: 2.6.3

  验证结果: ✅ 15 通过 | ⚠️ 2 警告 | ❌ 0 失败
```

**如何解读结果**:
- ✅ **PASS**: 该项检查通过
- ⚠️ **WARN**: 可选项未安装，不影响核心功能
- ❌ **FAIL**: 必须修复的问题，按照提示的命令安装即可

---

## 7. GPU 性能测试

```bash
python gpu_benchmark.py
```

本脚本测试三个关键指标：

### 7.1 矩阵乘法吞吐量 (TFLOPS)

衡量 GPU 的**计算能力**。大模型训练中，几乎所有计算都是矩阵乘法（注意力的 Q×K^T，FFN 的 x×W 等）。

**预期结果**（4090 参考值）:
```
Shape                     fp32       fp16       bf16
(1024×1024)×(1024×1024)    35.2       98.5       95.1
(4096×4096)×(4096×4096)    62.3      142.8      138.5
(8192×8192)×(8192×8192)    71.5      155.2      149.7
```

### 7.2 显存带宽 (GB/s)

衡量数据在 GPU 显存和计算单元之间的**传输速度**。大模型推理通常是"内存受限"的。

### 7.3 Flash Attention 加速比

对比标准 Attention 和 Flash Attention 的性能差异，通常有 2-4x 加速。

---

## 8. 显存估算实操

```bash
# 默认: 估算 0.5B 模型
python vram_estimator.py

# 估算 7B 模型
python vram_estimator.py --params 7.0 --dtype bf16 --zero 2

# 估算不同配置
python vram_estimator.py --params 0.5 --batch-size 8 --seq-length 1024
```

### 8.1 估算结果解读

以 Qwen2.5-0.5B + bf16 + ZeRO-2 为例：

```
参数显存:            0.93 GB    ← 模型权重本身
梯度显存:            1.86 GB    ← 反向传播的梯度 (fp32)
优化器显存:          3.73 GB    ← AdamW 的 m 和 v (fp32)
激活值显存:          0.82 GB    ← 前向传播中间结果 (含 gradient checkpointing)
框架开销:            1.50 GB    ← CUDA context + 临时 buffer
─────────────────────────
📊 总计:             8.84 GB    ← 24G 显存完全足够！
```

### 8.2 关键发现

运行后你会发现几个重要规律：

1. **优化器状态是最大开销**: AdamW 的 m 和 v 用 fp32 存储，占参数大小的 4 倍
2. **bf16 vs fp32 差距巨大**: fp32 训练 0.5B 模型需要 ~20GB，bf16 只需 ~9GB
3. **ZeRO Stage 在单卡上作用有限**: Stage 2 和 Stage 3 在单卡上收益不大
4. **Gradient Checkpointing 有效**: 开启后激活值显存大幅降低

---

## 9. 基线能力测评

```bash
# 测评 base 模型
python baseline_eval.py

# 测评 chat 模型（对比）
python baseline_eval.py --model Qwen/Qwen2.5-0.5B-Instruct
```

基线测评使用一组固定的 prompt 覆盖 5 个维度，评测结果保存为 JSON 文件，后续 p02-p05 每个阶段训练后都会用同样的 prompt 测评，形成对比。

### 预期观察

Base 模型（未经 SFT）的典型表现：

| 维度 | 表现 | 说明 |
|------|------|------|
| 中文知识 | 能续写，但可能跑题 | 会续写出像样的文本，但不一定回答问题 |
| 对话能力 | 很差 | 不会"回答问题"，只会续写 |
| 数学推理 | 很差 | 几乎无法做数学题 |
| 代码生成 | 一般 | 能续写一些代码片段 |
| 安全性 | 几乎没有 | 可能直接生成不当内容 |

> 📝 **说明**: 这些"缺点"正是后续 SFT (p03)、DPO (p04)、RL (p05) 要解决的问题。记住这些基线表现，后续对比时你会看到明显的进步。

---

## 10. 交互式推理体验

```bash
# 体验 base 模型（续写模式）
python inference.py

# 体验 Instruct 模型（对话模式）
python inference.py --chat

# 交互式对话
python inference.py --interactive
python inference.py --chat --interactive
```

### 10.1 对比体验

强烈建议同时体验 base 和 chat 两个版本：

**Base 模型**（`python inference.py`）：
```
输入: 你好，请问你是谁？
输出: 你好，请问你是谁？我是来自……（续写模式，不会回答问题）
```

**Chat 模型**（`python inference.py --chat`）：
```
输入: 你好，请问你是谁？
输出: 你好！我是 Qwen，一个AI助手。有什么可以帮助你的吗？
```

这个对比直观展示了 SFT 的核心作用：**让模型从"续写"变为"对话"**。

### 10.2 生成参数调节

在交互模式下，你可以感受不同参数的影响：

| 参数 | 低值效果 | 高值效果 | 推荐值 |
|------|---------|---------|--------|
| `temperature` | 确定性强，但缺乏多样性 | 随机性强，可能胡说 | 0.7 |
| `top_p` | 只选最可能的 token | 候选范围更大 | 0.9 |
| `top_k` | 候选少，更确定 | 候选多，更随机 | 50 |

---

## 小结

完成本模块后，你已经：

- [x] 在云服务器上搭建了完整的训练环境
- [x] 下载了 Qwen2.5-0.5B 基座模型
- [x] 理解了 GPU 显存的构成和估算方法
- [x] 测试了 GPU 的实际性能
- [x] 记录了原始模型的能力基线
- [x] 体验了 base vs chat 模型的差异

**下一步**: 进入 [p02 继续预训练](../p02_continual_pretrain/README.md)，让模型学习更多中文知识。
