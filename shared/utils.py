"""
公共工具函数：日志记录、指标计算、Checkpoint 管理、可视化等。
所有版本均可复用这些工具。
"""

import os
import random
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. 随机种子设置 —— 保证实验可复现
# ============================================================
def set_seed(seed: int = 42):
    """
    统一设置所有随机种子，确保实验可复现。
    
    Args:
        seed: 随机种子值，默认 42
    
    为什么需要设置多个随机源？
    - random: Python 内置随机模块，影响数据采样等
    - numpy: 影响数据预处理中的随机操作
    - torch: 影响模型初始化、Dropout 等
    - cuda: 影响 GPU 上的随机操作
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 场景
        # 确保 cuDNN 确定性（会略微降低速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 2. 日志记录器 —— 统一日志格式
# ============================================================
def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    创建一个格式统一的 Logger。
    
    Args:
        name: Logger 名称（通常用模块名）
        log_file: 可选的日志文件路径
        level: 日志级别
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 3. Checkpoint 管理 —— 模型保存与加载
# ============================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **extra_info
):
    """
    保存训练 Checkpoint。
    
    为什么要保存这些信息？
    - model state_dict: 模型参数（最核心）
    - optimizer state_dict: 优化器状态（如 Adam 的动量），用于断点续训
    - epoch: 当前训练轮次
    - loss: 当前 loss 值，便于判断训练状态
    - extra_info: 其他自定义信息（如 best_metric、config 等）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **extra_info,
    }
    
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] 已保存到 {save_path} (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    加载训练 Checkpoint。
    
    Args:
        model: 要加载权重的模型
        save_path: checkpoint 文件路径
        optimizer: 可选的优化器（如需断点续训则传入）
        device: 加载到哪个设备
    
    Returns:
        包含 epoch、loss 等额外信息的字典
    """
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"[Checkpoint] 已从 {save_path} 加载 (epoch={checkpoint.get('epoch', '?')})")
    
    return {k: v for k, v in checkpoint.items() 
            if k not in ("model_state_dict", "optimizer_state_dict")}


# ============================================================
# 4. 模型参数统计
# ============================================================
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量。
    
    Args:
        model: PyTorch 模型
        trainable_only: 是否仅统计可训练参数
    
    Returns:
        参数总数
    
    为什么要区分可训练/全部参数？
    - LoRA 等参数高效微调方法会冻结大部分参数
    - 了解可训练参数量有助于估算显存和计算需求
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())
    return total


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """打印模型参数摘要信息。"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*50}")
    print(f" {model_name} 参数摘要")
    print(f"{'='*50}")
    print(f" 总参数量:     {total_params:>12,}")
    print(f" 可训练参数:   {trainable_params:>12,}")
    print(f" 冻结参数:     {frozen_params:>12,}")
    print(f" 可训练占比:   {trainable_params/max(total_params,1)*100:>11.2f}%")
    print(f"{'='*50}\n")


# ============================================================
# 5. 设备管理
# ============================================================
def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    自动选择最佳可用设备。
    
    优先级: CUDA GPU > Apple MPS > CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] 使用 Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] 使用 CPU")
    return device


# ============================================================
# 6. 训练指标追踪器
# ============================================================
class AverageMeter:
    """
    滑动平均计算器，用于追踪 Loss、准确率等指标。
    
    使用方式:
        meter = AverageMeter("loss")
        for batch in dataloader:
            loss = ...
            meter.update(loss.item(), batch_size)
        print(f"平均 loss: {meter.avg:.4f}")
    """
    
    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0     # 最近一次的值
        self.avg = 0     # 全局平均
        self.sum = 0     # 总和
        self.count = 0   # 样本数
    
    def update(self, val: float, n: int = 1):
        """
        更新指标。
        
        Args:
            val: 当前批次的指标值
            n: 当前批次的样本数（用于加权平均）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


# ============================================================
# 7. 早停机制
# ============================================================
class EarlyStopping:
    """
    早停机制：当验证指标连续 patience 轮没有改善时停止训练。
    
    为什么需要早停？
    - 防止过拟合：当模型在验证集上的表现不再提升，继续训练只会拟合噪声
    - 节省计算资源：避免无意义的训练轮次
    
    使用方式:
        early_stop = EarlyStopping(patience=5)
        for epoch in range(max_epochs):
            val_loss = validate(...)
            if early_stop(val_loss):
                print("触发早停!")
                break
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: 允许的连续无改善轮数
            min_delta: 最小改善量（低于此值不算改善）
            mode: "min"（loss 越小越好）或 "max"（acc 越大越好）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


# ============================================================
# 8. 训练曲线可视化
# ============================================================
def plot_training_curves(
    train_losses: list,
    val_losses: Optional[list] = None,
    train_metrics: Optional[list] = None,
    val_metrics: Optional[list] = None,
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
    title: str = "Training Curves",
):
    """
    绘制训练曲线图（Loss 和指标）。
    
    Args:
        train_losses: 每个 epoch 的训练 loss
        val_losses: 每个 epoch 的验证 loss
        train_metrics: 每个 epoch 的训练指标
        val_metrics: 每个 epoch 的验证指标
        metric_name: 指标名称
        save_path: 图片保存路径
        title: 图标题
    """
    has_metrics = train_metrics is not None or val_metrics is not None
    fig, axes = plt.subplots(1, 2 if has_metrics else 1, figsize=(12 if has_metrics else 6, 5))
    
    if not has_metrics:
        axes = [axes]
    
    # Loss 曲线
    axes[0].plot(train_losses, label="Train Loss", color="blue")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 指标曲线
    if has_metrics:
        if train_metrics:
            axes[1].plot(train_metrics, label=f"Train {metric_name}", color="blue")
        if val_metrics:
            axes[1].plot(val_metrics, label=f"Val {metric_name}", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f"{title} - {metric_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] 训练曲线已保存到 {save_path}")
    
    plt.close()


# ============================================================
# 9. 计时工具
# ============================================================
class Timer:
    """简单计时器，用于衡量训练/推理耗时。"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def __str__(self):
        if self.elapsed < 60:
            return f"{self.elapsed:.2f}s"
        elif self.elapsed < 3600:
            return f"{self.elapsed/60:.1f}min"
        else:
            return f"{self.elapsed/3600:.1f}h"
