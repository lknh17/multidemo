"""v13 高级注意力机制 - 训练: MHA/GQA/MQA 对比实验"""
import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AdvancedAttentionConfig, config
from model import AdvancedTransformer
from dataset import create_dataloaders, save_demo_data
from shared.utils import (
    set_seed, get_device, save_checkpoint, AverageMeter,
    get_logger, print_model_summary,
)


def train_variant(name, n_heads, n_kv_heads, cfg, device, logger):
    """训练某种注意力变体并返回结果"""
    # 创建配置
    variant_cfg = AdvancedAttentionConfig(
        d_model=cfg.d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        n_layers=cfg.n_layers, d_ff=cfg.d_ff, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs, weight_decay=cfg.weight_decay,
    )
    
    train_loader, val_loader = create_dataloaders(variant_cfg)
    model = AdvancedTransformer(variant_cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    kv_cache_bytes = model.count_kv_cache_size(
        variant_cfg.n_layers, n_kv_heads,
        variant_cfg.d_model // n_heads, variant_cfg.max_seq_len
    )
    
    logger.info(f"[{name}] 参数量: {total_params:,} | KV Cache: {kv_cache_bytes/1024:.1f}KB")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=variant_cfg.learning_rate,
        weight_decay=variant_cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(1, variant_cfg.num_epochs + 1):
        model.train()
        loss_meter = AverageMeter("loss")
        
        start_time = time.time()
        for batch in tqdm(train_loader, desc=f"[{name}] E{epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, variant_cfg.vocab_size), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), variant_cfg.max_grad_norm)
            optimizer.step()
            
            loss_meter.update(loss.item(), input_ids.size(0))
        
        epoch_time = time.time() - start_time
        
        # 验证
        model.eval()
        val_meter = AverageMeter("val_loss")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, variant_cfg.vocab_size), labels.view(-1))
                val_meter.update(loss.item(), input_ids.size(0))
        
        logger.info(
            f"[{name}] Epoch {epoch} | Train: {loss_meter.avg:.4f} | "
            f"Val: {val_meter.avg:.4f} | Time: {epoch_time:.1f}s"
        )
        
        if val_meter.avg < best_val_loss:
            best_val_loss = val_meter.avg
            save_checkpoint(
                model, optimizer, epoch, loss_meter.avg,
                os.path.join(config.checkpoint_dir, f"{name}_best.pt"),
                val_loss=val_meter.avg,
            )
    
    return best_val_loss


def main():
    set_seed(42)
    device = get_device()
    logger = get_logger("v13")
    save_demo_data()
    
    logger.info("=" * 60)
    logger.info("v13 高级注意力机制 - MHA/GQA/MQA 对比实验")
    logger.info("=" * 60)
    
    # 对比实验：不同注意力变体
    variants = [
        ("MHA", config.n_heads, config.n_heads),      # 标准多头
        ("GQA", config.n_heads, config.n_kv_heads),    # 分组查询
        ("MQA", config.n_heads, 1),                     # 多查询
    ]
    
    results = {}
    for name, n_heads, n_kv_heads in variants:
        logger.info(f"\n{'='*40} {name} {'='*40}")
        val_loss = train_variant(name, n_heads, n_kv_heads, config, device, logger)
        results[name] = val_loss
    
    # 打印对比结果
    logger.info("\n" + "=" * 60)
    logger.info("对比结果：")
    for name, val_loss in results.items():
        logger.info(f"  {name}: best_val_loss = {val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
