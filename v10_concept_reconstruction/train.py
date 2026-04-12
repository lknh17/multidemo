"""v10 联合训练: 对比 + 重构 + DeepStack"""
import os, sys, torch, torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from config import config
from model import ConceptModel, ConceptReconLoss
from dataset import create_dataloaders, save_demo_data
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def infonce(a, b, s):
    logits = s * a @ b.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

def main():
    set_seed(42); device = get_device(); logger = get_logger("v10")
    save_demo_data()
    train_loader, _ = create_dataloaders(config)
    model = ConceptModel(config).to(device)
    print_model_summary(model, "ConceptModel")
    recon_loss_fn = ConceptReconLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); meters = {k: AverageMeter(k) for k in ["total", "contrastive", "recon"]}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            out = model(batch["image"].to(device), batch["input_ids"].to(device))
            targets = {k: batch[k].to(device) for k in ["industry", "brand", "attributes", "intent"]}
            
            cl = infonce(out["fused_img"], out["fused_txt"], out["logit_scale"])
            rl_img, _ = recon_loss_fn(out["img_concepts"], targets)
            rl_txt, _ = recon_loss_fn(out["txt_concepts"], targets)
            rl = (rl_img + rl_txt) / 2
            
            # Uncertainty weighting
            s_c, s_r = out["log_sigma_c"], out["log_sigma_r"]
            total = (1/(2*s_c.exp()**2)) * cl + s_c + (1/(2*s_r.exp()**2)) * rl + s_r
            
            optimizer.zero_grad(); total.backward(); optimizer.step()
            meters["total"].update(total.item()); meters["contrastive"].update(cl.item()); meters["recon"].update(rl.item())
        
        info = " | ".join(f"{k}: {m.avg:.4f}" for k, m in meters.items())
        logger.info(f"Epoch {epoch} | {info}")
    save_checkpoint(model, optimizer, epoch, meters["total"].avg, os.path.join(config.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
