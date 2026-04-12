"""v02 ViT 推理 + Attention Map 可视化"""
import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from model import ViT
from dataset import create_dataloaders, CIFAR10_CLASSES, get_transforms
from shared.utils import get_device, load_checkpoint

@torch.no_grad()
def predict(model, images, device):
    model.eval()
    images = images.to(device)
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return preds, probs

def visualize_predictions(model, val_loader, device, n=8):
    images, labels = next(iter(val_loader))
    images, labels = images[:n], labels[:n]
    preds, probs = predict(model, images, device)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
    
    fig, axes = plt.subplots(2, n//2, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"Pred:{CIFAR10_CLASSES[preds[i]]}\nTrue:{CIFAR10_CLASSES[labels[i]]}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/predictions.png", dpi=150)
    plt.close()
    print("Predictions saved to logs/predictions.png")

def main():
    device = get_device()
    model = ViT(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt):
        load_checkpoint(model, ckpt, device=str(device))
    else:
        print("No checkpoint found. Run train.py first.")
    _, val_loader = create_dataloaders(config)
    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()
