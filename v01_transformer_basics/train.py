"""
v01 Transformer 基础 - 训练脚本

本脚本实现完整的 Transformer 训练流程：
1. 数据加载与预处理
2. 模型初始化
3. 学习率调度（Warmup + Cosine Decay）
4. 训练循环（Teacher Forcing）
5. 验证与评估
6. Checkpoint 保存
7. 训练曲线可视化

运行方式:
    cd v01_transformer_basics
    python train.py
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 将项目根目录加入 sys.path，以便导入 shared 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TransformerConfig, config
from model import MiniTransformer
from dataset import create_dataloaders, save_demo_data
from shared.utils import (
    set_seed, get_logger, save_checkpoint, count_parameters,
    get_device, AverageMeter, plot_training_curves, print_model_summary,
)


# ============================================================
# 1. 学习率调度器：Warmup + Cosine Decay
# ============================================================
class WarmupCosineScheduler:
    """
    学习率调度：先线性 warmup，然后余弦衰减。
    
    为什么需要 Warmup？
    - 训练初期，模型参数是随机的，梯度方向不可靠
    - 如果一开始就用大学习率，可能导致训练不稳定（loss 爆炸）
    - Warmup 让学习率从 0 缓慢增长到目标值，给模型一个"热身"的过程
    
    为什么用余弦衰减？
    - 训练后期需要更小的学习率来精细调整参数
    - 余弦衰减比线性衰减更平滑，效果通常更好
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """更新学习率。每个训练 step 调用一次。"""
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def _compute_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            # Warmup 阶段：学习率从 0 线性增长到 base_lr
            return self.base_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine Decay 阶段：从 base_lr 余弦衰减到 min_lr
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# 2. 训练一个 Epoch
# ============================================================
def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    criterion: nn.Module,
    device: torch.device,
    config: TransformerConfig,
) -> float:
    """
    训练一个完整的 epoch。
    
    训练策略：Teacher Forcing
    - 解码器在每步接收的输入是"真实的"前一个 token（而不是模型上一步的预测）
    - 好处：训练更稳定，收敛更快
    - 缺点：训练和推理时的输入分布不同（Exposure Bias）
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # 将数据移到设备上
        src = batch["src"].to(device)            # [B, Ls]
        tgt_input = batch["tgt_input"].to(device)  # [B, Lt] (BOS + sorted)
        tgt_output = batch["tgt_output"].to(device) # [B, Lt] (sorted + EOS)
        
        # 创建掩码
        # 因果掩码：确保解码器位置 i 只能看到 0..i
        tgt_mask = MiniTransformer.generate_causal_mask(
            tgt_input.size(1), device
        )
        
        # 前向传播
        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        # logits: [B, Lt, vocab_size]
        
        # 计算 Loss
        # 将 logits 和 target 展平，因为 CrossEntropyLoss 期望的输入是 [N, C] 和 [N]
        loss = criterion(
            logits.view(-1, config.vocab_size),  # [B*Lt, vocab_size]
            tgt_output.view(-1),                  # [B*Lt]
        )
        
        # 反向传播
        optimizer.zero_grad()  # 清空上一步的梯度
        loss.backward()        # 计算梯度
        
        # 梯度裁剪：防止梯度爆炸
        # 当梯度的全局 L2 范数超过 max_grad_norm 时，按比例缩小所有梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()   # 更新参数
        scheduler.step()   # 更新学习率
        
        loss_meter.update(loss.item(), src.size(0))
    
    return loss_meter.avg


# ============================================================
# 3. 验证
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    config: TransformerConfig,
) -> dict:
    """
    在验证集上评估模型。
    
    除了 Loss，还计算：
    - Token 级准确率：每个位置预测正确的比例
    - 序列级准确率：整个序列完全正确的比例
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        
        tgt_mask = MiniTransformer.generate_causal_mask(
            tgt_input.size(1), device
        )
        
        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        
        loss = criterion(
            logits.view(-1, config.vocab_size),
            tgt_output.view(-1),
        )
        loss_meter.update(loss.item(), src.size(0))
        
        # 取概率最高的 token 作为预测
        predictions = logits.argmax(dim=-1)  # [B, Lt]
        
        # Token 级准确率
        correct_tokens += (predictions == tgt_output).sum().item()
        total_tokens += tgt_output.numel()
        
        # 序列级准确率（整个序列完全匹配才算对）
        seq_correct = (predictions == tgt_output).all(dim=-1)
        correct_sequences += seq_correct.sum().item()
        total_sequences += src.size(0)
    
    token_acc = correct_tokens / max(total_tokens, 1)
    seq_acc = correct_sequences / max(total_sequences, 1)
    
    return {
        "val_loss": loss_meter.avg,
        "token_accuracy": token_acc,
        "sequence_accuracy": seq_acc,
    }


# ============================================================
# 4. 主训练流程
# ============================================================
def main():
    # ---- 初始化 ----
    set_seed(42)
    logger = get_logger("v01_train")
    device = get_device()
    
    logger.info("=" * 60)
    logger.info("v01 Transformer 基础 - 序列排序训练")
    logger.info("=" * 60)
    
    # ---- 保存 Demo 数据 ----
    save_demo_data(config)
    
    # ---- 数据 ----
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批")
    
    # ---- 模型 ----
    model = MiniTransformer(config).to(device)
    print_model_summary(model, "Mini Transformer")
    
    # ---- 损失函数 ----
    # CrossEntropyLoss: 交叉熵损失，衡量预测分布与真实分布的差异
    # ignore_index=0: 忽略 PAD token 的 loss（PAD 位置不参与损失计算）
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # ---- 优化器 ----
    # AdamW: Adam + Weight Decay
    # 为什么用 AdamW 而不是 Adam？
    # - 标准 Adam 的 L2 正则化实现有误（与自适应学习率耦合）
    # - AdamW 解耦了权重衰减，正则化效果更好
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),  # beta2=0.98 是 Transformer 论文推荐的值
    )
    
    # ---- 学习率调度 ----
    total_steps = len(train_loader) * config.num_epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        base_lr=config.learning_rate,
    )
    
    # ---- 训练循环 ----
    train_losses = []
    val_losses = []
    val_token_accs = []
    val_seq_accs = []
    best_val_loss = float("inf")
    
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.num_epochs} | LR: {scheduler.get_lr():.6f}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config
        )
        train_losses.append(train_loss)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, config)
        val_losses.append(val_metrics["val_loss"])
        val_token_accs.append(val_metrics["token_accuracy"])
        val_seq_accs.append(val_metrics["sequence_accuracy"])
        
        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Token Acc: {val_metrics['token_accuracy']:.4f} | "
            f"Seq Acc: {val_metrics['sequence_accuracy']:.4f}"
        )
        
        # 保存最佳模型
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config.checkpoint_dir, "best_model.pt"),
                val_loss=val_metrics["val_loss"],
                token_accuracy=val_metrics["token_accuracy"],
                sequence_accuracy=val_metrics["sequence_accuracy"],
            )
        
        # 每 10 个 epoch 保存一次
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config.checkpoint_dir, f"epoch_{epoch}.pt"),
            )
    
    # ---- 保存训练曲线 ----
    plot_training_curves(
        train_losses, val_losses,
        train_metrics=val_token_accs,
        val_metrics=val_seq_accs,
        metric_name="Accuracy",
        save_path=os.path.join(config.log_dir, "training_curves.png"),
        title="Transformer Sorting",
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳验证 Loss: {best_val_loss:.4f}")
    logger.info(f"最终序列准确率: {val_seq_accs[-1]:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
