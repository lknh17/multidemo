"""v09 DeepStack 训练"""
import os, sys, torch, torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import DeepStackModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v08_contrastive_learning"))
from dataset import AdRelevanceDataset
from torch.utils.data import DataLoader
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def infonce(a, b, scale):
    logits = scale * a @ b.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

def main():
    set_seed(42); device = get_device(); logger = get_logger("v09")
    ds = AdRelevanceDataset(2000)
    train_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    model = DeepStackModel(config).to(device)
    print_model_summary(model, "DeepStack")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); total_m, layer_m, fuse_m = AverageMeter("total"), AverageMeter("layer"), AverageMeter("fuse")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            out = model(batch["image"].to(device), batch["input_ids"].to(device))
            s = out["logit_scale"]
            # 各层 Loss
            layer_loss = sum(infonce(ie, te, s) for ie, te in zip(out["layer_img_embs"], out["layer_txt_embs"])) / len(out["layer_img_embs"])
            # 融合 Loss
            fuse_loss = infonce(out["fused_img"], out["fused_txt"], s)
            total = config.layer_loss_weight * layer_loss + config.fusion_loss_weight * fuse_loss
            optimizer.zero_grad(); total.backward(); optimizer.step()
            total_m.update(total.item()); layer_m.update(layer_loss.item()); fuse_m.update(fuse_loss.item())
        logger.info(f"Epoch {epoch} | Total: {total_m.avg:.4f} | Layer: {layer_m.avg:.4f} | Fuse: {fuse_m.avg:.4f}")
    save_checkpoint(model, optimizer, epoch, total_m.avg, os.path.join(config.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
