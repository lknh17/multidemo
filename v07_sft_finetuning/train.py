"""v07 SFT 训练: LoRA 微调"""
import os, sys, torch, torch.nn as nn
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import SFTModel, inject_lora
from dataset import create_dataloaders, save_demo_data
from shared.utils import set_seed, get_device, save_checkpoint, AverageMeter, get_logger, print_model_summary

def main():
    set_seed(42); device = get_device(); logger = get_logger("v07")
    save_demo_data()
    train_loader, val_loader = create_dataloaders(config)
    model = SFTModel(config)
    model = inject_lora(model, r=config.lora_r, alpha=config.lora_alpha)
    model = model.to(device)
    print_model_summary(model, "SFT+LoRA")
    
    # 只优化可训练参数（LoRA 参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train(); lm = AverageMeter("loss")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, ids, labels = batch["image"].to(device), batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(imgs, ids)
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lm.update(loss.item())
        logger.info(f"Epoch {epoch} | Loss: {lm.avg:.4f}")
    save_checkpoint(model, optimizer, config.num_epochs, lm.avg, os.path.join(config.checkpoint_dir, "lora_model.pt"))

if __name__ == "__main__":
    main()
