"""
v12 FSDP 训练脚本 (PyTorch 原生分布式)

运行: torchrun --nproc_per_node=4 train_fsdp.py
"""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v11_multimodal_embedding"))

from config import config
from shared.utils import set_seed, get_logger, AverageMeter
from model import EmbeddingModel as BaseModel
from dataset import RetrievalDataset


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False


def main():
    rank, world_size, distributed = setup_distributed()
    set_seed(42 + rank)
    logger = get_logger(f"v12_fsdp_rank{rank}")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    train_ds = RetrievalDataset(2000)
    sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                              shuffle=(sampler is None), drop_last=True)
    
    model_cfg = type("Cfg", (), {k: v for k, v in vars(config).items()})()
    model = BaseModel(model_cfg).to(device)
    
    if distributed:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(model)
            logger.info("Using FSDP")
        except ImportError:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            logger.info("FSDP not available, using DDP")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(1, config.num_epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        lm = AverageMeter("loss"); t0 = time.time()
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False, disable=rank != 0):
            ie, te, s = model(batch["image"].to(device), batch["input_ids"].to(device))
            logits = s * ie @ te.t()
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lm.update(loss.item())
        if rank == 0:
            logger.info(f"Epoch {epoch} | Loss: {lm.avg:.4f} | Time: {time.time()-t0:.1f}s")
    
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
