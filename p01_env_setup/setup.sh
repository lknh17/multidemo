#!/bin/bash
# ============================================================
# p01 环境搭建 - 一键安装脚本
#
# 本脚本安装大模型实践全流程所需的所有依赖。
# 每个包都附有注释说明其用途，便于理解和排查问题。
#
# 使用方式:
#   chmod +x setup.sh
#   bash setup.sh
#
# 前置条件:
#   - Ubuntu 20.04+ / CentOS 7+
#   - Python 3.10+
#   - CUDA 11.8+ 和对应的 NVIDIA Driver
#   - 建议使用 conda 或 venv 虚拟环境
# ============================================================

set -e  # 遇到错误立即退出

# ---- 0. 配置 python 和 pip 别名 ----
# 确保 python 命令指向 python3，pip 命令指向 pip3
shopt -s expand_aliases
alias python=python3
alias pip=pip3
echo "[Pre-check] 已配置 python→python3, pip→pip3 别名"

echo "=================================================="
echo "  实践大模型 - 环境一键安装"
echo "=================================================="

# ---- 0. 检查 Python 版本 ----
echo ""
echo "[Step 0] 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  当前 Python: $python_version"

# ---- 1. 升级 pip ----
echo ""
echo "[Step 1] 升级 pip..."
pip install --upgrade pip

# ---- 2. 安装 PyTorch (CUDA 11.8) ----
# PyTorch 是深度学习的基础框架，所有训练和推理都基于它
# 这里安装 CUDA 11.8 版本，兼容大多数 GPU 和 Driver
echo ""
echo "[Step 2] 安装 PyTorch..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- 3. 安装 HuggingFace 生态 ----
# transformers: 模型加载和训练的核心库
# datasets: 数据集加载和预处理
# accelerate: 分布式训练和混合精度训练的封装
# peft: 参数高效微调（LoRA/QLoRA/DoRA）
# trl: 强化学习训练（GRPO/DPO/PPO）
# tokenizers: 高速 tokenizer 实现
echo ""
echo "[Step 3] 安装 HuggingFace 生态..."
pip install \
    transformers>=4.45.0 \
    datasets>=2.20.0 \
    accelerate>=0.34.0 \
    peft>=0.12.0 \
    trl>=0.11.0 \
    tokenizers>=0.20.0 \
    huggingface_hub>=0.25.0 \
    safetensors>=0.4.0

# ---- 4. 安装 DeepSpeed ----
# DeepSpeed: 微软开源的分布式训练框架
# ZeRO 优化可以大幅降低单卡显存占用
# 安装需要编译 C++/CUDA 扩展，可能需要几分钟
echo ""
echo "[Step 4] 安装 DeepSpeed..."
pip install deepspeed>=0.15.0

# ---- 5. 安装 Flash Attention 2 ----
# Flash Attention: 高效的注意力计算实现
# 比标准 Attention 快 2-4x，显存节省 5-20x
# 需要 Ampere+ GPU (A100/A6000/4090 等)
# 安装可能需要 10-15 分钟（需要编译 CUDA 内核）
echo ""
echo "[Step 5] 安装 Flash Attention 2..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "  ⚠️  Flash Attention 安装失败（可能是 GPU 不支持或编译工具缺失）"
    echo "  训练仍可运行，但速度会稍慢"
}

# ---- 6. 安装量化工具 ----
# bitsandbytes: 8-bit/4-bit 量化训练和推理（QLoRA 的基础）
# auto-gptq: GPTQ 量化方法
# autoawq: AWQ 量化方法
echo ""
echo "[Step 6] 安装量化工具..."
pip install \
    bitsandbytes>=0.43.0 \
    auto-gptq>=0.7.0 \
    autoawq>=0.2.0

# ---- 7. 安装推理框架 ----
# vllm: 高性能推理引擎（PagedAttention）
# sglang: 另一个高性能推理框架
echo ""
echo "[Step 7] 安装推理框架..."
pip install vllm>=0.6.0 2>/dev/null || {
    echo "  ⚠️  vLLM 安装失败，将在 p08 模块单独处理"
}

# ---- 8. 安装评测工具 ----
# lm-eval: 标准化的 LLM 评测框架
echo ""
echo "[Step 8] 安装评测工具..."
pip install lm-eval>=0.4.0 2>/dev/null || {
    echo "  ⚠️  lm-eval 安装失败，将在 p09 模块单独处理"
}

# ---- 9. 安装通用工具 ----
# wandb: 训练监控和实验追踪（可选）
# matplotlib: 绘图（训练曲线、对比图表）
# tqdm: 进度条
# scipy: 科学计算（评测时需要）
# sentencepiece: 某些 tokenizer 的依赖
echo ""
echo "[Step 9] 安装通用工具..."
pip install \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    sentencepiece>=0.2.0 \
    wandb>=0.16.0 \
    pandas>=2.0.0 \
    tabulate>=0.9.0

# ---- 10. 验证安装 ----
echo ""
echo "[Step 10] 验证核心依赖..."
python3 -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA 可用:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  显存:          {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
    print(f'  CUDA 版本:     {torch.version.cuda}')
    print(f'  cuDNN 版本:    {torch.backends.cudnn.version()}')

import transformers
print(f'  Transformers:  {transformers.__version__}')

import peft
print(f'  PEFT:          {peft.__version__}')

import trl
print(f'  TRL:           {trl.__version__}')

import deepspeed
print(f'  DeepSpeed:     {deepspeed.__version__}')

try:
    import flash_attn
    print(f'  Flash-Attn:    {flash_attn.__version__}')
except ImportError:
    print(f'  Flash-Attn:    未安装（可选）')

try:
    import bitsandbytes
    print(f'  BitsAndBytes:  {bitsandbytes.__version__}')
except ImportError:
    print(f'  BitsAndBytes:  未安装（可选）')
"

echo ""
echo "=================================================="
echo "  ✅ 环境安装完成！"
echo ""
echo "  下一步："
echo "    1. 运行 python verify_env.py 进行详细验证"
echo "    2. 运行 python download_model.py 下载模型"
echo "    3. 运行 python gpu_benchmark.py 测试 GPU 性能"
echo "=================================================="
