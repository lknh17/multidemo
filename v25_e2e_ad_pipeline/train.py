"""
V25 - 端到端广告管线训练脚本
==============================
python train.py --mode pipeline
python train.py --mode ctr
python train.py --mode ranker
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter
from config import FullConfig
from model import E2EAdPipeline, CTRPredictor, MultiObjectiveRanker
from dataset import create_ad_dataloaders


def train_pipeline(config, logger):
    """模式 1：端到端管线训练（对比学习 + 安全分类）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("E2E Ad Pipeline Training (Contrastive)")
    logger.info("=" * 60)

    train_loader, val_loader = create_ad_dataloaders(config)
    model = E2EAdPipeline(config).to(device)
    logger.info(f"Pipeline params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            text_ids = batch['text_ids'].to(device)
            audio = batch['audio_feats'].to(device) if config.creative.has_audio else None

            # 自对比：同 batch 内正对角线为正例
            outputs = model(images, text_ids, images, text_ids,
                            query_audio=audio, ad_audio=audio)

            loss = outputs['contrastive_loss']

            # 安全分类辅助损失
            if outputs['safety_scores'] is not None:
                safety_target = torch.zeros_like(outputs['safety_scores'])
                safety_loss = nn.functional.binary_cross_entropy(
                    outputs['safety_scores'], safety_target)
                loss = loss + 0.1 * safety_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def train_ctr(config, logger):
    """模式 2：CTR 预估训练"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("CTR Predictor Training (DeepFM)")
    logger.info("=" * 60)

    train_loader, _ = create_ad_dataloaders(config)
    D = config.creative.d_model
    model = CTRPredictor(D).to(device)
    logger.info(f"CTR model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user_emb = batch['user_emb'].to(device)
            # 用 context_emb 作为广告 embedding 替代（简化）
            ad_emb = torch.randn(user_emb.shape[0], D).to(device) * 0.1
            context_emb = batch['context_emb'].to(device)
            clicked = batch['clicked'].to(device)

            pred_ctr = model(user_emb, ad_emb, context_emb)
            loss = nn.functional.binary_cross_entropy(pred_ctr, clicked)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())

        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")


def train_ranker(config, logger):
    """模式 3：多目标排序训练"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Multi-Objective Ranker Training")
    logger.info("=" * 60)

    train_loader, _ = create_ad_dataloaders(config)
    D = config.creative.d_model
    model = MultiObjectiveRanker(D, config).to(device)
    logger.info(f"Ranker params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user_emb = batch['user_emb'].to(device)
            ad_emb = torch.randn(user_emb.shape[0], D).to(device) * 0.1
            context_emb = batch['context_emb'].to(device)
            clicked = batch['clicked'].to(device)
            relevance = batch['relevance'].to(device)

            outputs = model(user_emb, ad_emb, context_emb)

            # 多目标损失
            ctr_loss = nn.functional.binary_cross_entropy(outputs['ctr_score'], clicked)
            rel_loss = nn.functional.mse_loss(outputs['relevance_score'], relevance)
            listwise_loss = model.compute_listwise_loss(
                outputs['final_score'].unsqueeze(0), relevance.unsqueeze(0))
            loss = ctr_loss + rel_loss + 0.5 * listwise_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())

        w = outputs['weights']
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}  "
                     f"weights=[ctr:{w[0]:.2f} rel:{w[1]:.2f} div:{w[2]:.2f} fresh:{w[3]:.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pipeline",
                        choices=["pipeline", "ctr", "ranker"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = FullConfig()
    if args.epochs:
        config.num_epochs = args.epochs

    set_seed(config.seed)
    logger = get_logger("V25-E2E-Ad")

    mode_fn = {
        "pipeline": train_pipeline,
        "ctr": train_ctr,
        "ranker": train_ranker,
    }
    mode_fn[args.mode](config, logger)


if __name__ == "__main__":
    main()
