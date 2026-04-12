"""v11 端到端训练"""
import os, sys, torch, torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from config import config
from model import EmbeddingModel
from dataset import create_dataloaders
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def main():
    set_seed(42); device = get_device(); logger = get_logger("v11")
    train_loader, _, _ = create_dataloaders(config)
    model = EmbeddingModel(config).to(device)
    print_model_summary(model, "EmbeddingModel")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); lm = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            ie, te, s = model(batch["image"].to(device), batch["input_ids"].to(device))
            logits = s * ie @ te.t()
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lm.update(loss.item())
        logger.info(f"Epoch {epoch} | Loss: {lm.avg:.4f}")
    save_checkpoint(model, optimizer, epoch, lm.avg, os.path.join(config.checkpoint_dir, "best.pt"))

if __name__ == "__main__":
    main()
