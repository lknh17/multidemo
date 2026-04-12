"""
v12 DeepSpeed 分布式训练脚本

运行方式:
  单卡: python train_deepspeed.py
  多卡: deepspeed --num_gpus=4 train_deepspeed.py
"""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v11_multimodal_embedding"))

from config import config
from shared.utils import set_seed, get_logger, AverageMeter, print_model_summary

# 复用 v11 模型和数据
from model import EmbeddingModel as BaseModel
from dataset import RetrievalDataset


def main():
    set_seed(42)
    logger = get_logger("v12_ds")
    
    try:
        import deepspeed
        USE_DS = True
        logger.info("DeepSpeed available, using distributed training")
    except ImportError:
        USE_DS = False
        logger.info("DeepSpeed not installed, falling back to single GPU with gradient accumulation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据
    train_ds = RetrievalDataset(2000)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    # 模型
    from config import DistributedConfig
    model_cfg = type("Cfg", (), {k: v for k, v in vars(config).items()})()
    model = BaseModel(model_cfg)
    print_model_summary(model, "Distributed Model")
    
    if USE_DS:
        # DeepSpeed 初始化
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=os.path.join(os.path.dirname(__file__), "ds_config.json"),
            model_parameters=model.parameters(),
        )
        device = model_engine.local_rank
    else:
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if (config.fp16 and torch.cuda.is_available()) else None
    
    # 训练
    for epoch in range(1, config.num_epochs + 1):
        lm = AverageMeter("loss")
        t0 = time.time()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            
            if USE_DS:
                ie, te, s = model_engine(imgs, ids)
                logits = s * ie @ te.t()
                labels = torch.arange(logits.size(0), device=logits.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
                model_engine.backward(loss)
                model_engine.step()
            else:
                # 梯度累积 + 混合精度
                with torch.amp.autocast("cuda", enabled=config.fp16 and torch.cuda.is_available()):
                    ie, te, s = model(imgs, ids)
                    logits = s * ie @ te.t()
                    labels = torch.arange(logits.size(0), device=logits.device)
                    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
                    loss = loss / config.gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            
            lm.update(loss.item() * config.gradient_accumulation_steps)
        
        elapsed = time.time() - t0
        throughput = len(train_loader.dataset) / elapsed
        logger.info(f"Epoch {epoch} | Loss: {lm.avg:.4f} | Time: {elapsed:.1f}s | Throughput: {throughput:.0f} samples/s")

if __name__ == "__main__":
    main()
