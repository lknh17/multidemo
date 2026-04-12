"""
v01 Transformer 基础 - 推理脚本

本脚本实现两种解码策略：
1. 贪心解码 (Greedy Decoding): 每步选概率最高的 token
2. Beam Search: 维护 beam_size 个候选序列，选全局最优

使用方式:
    cd v01_transformer_basics
    python inference.py
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TransformerConfig, config
from model import MiniTransformer
from shared.utils import get_device, load_checkpoint


# ============================================================
# 1. 贪心解码
# ============================================================
@torch.no_grad()
def greedy_decode(
    model: MiniTransformer,
    src: torch.Tensor,         # [1, Ls] 源序列
    max_len: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> list:
    """
    贪心解码：每步选择概率最高的 token。
    
    优点：简单、快速
    缺点：局部最优，可能不是全局最优解
    
    解码过程（以排序 [5, 2, 8] 为例）：
    Step 0: 输入 [BOS]        → 预测 2  (概率最高)
    Step 1: 输入 [BOS, 2]     → 预测 5
    Step 2: 输入 [BOS, 2, 5]  → 预测 8
    Step 3: 输入 [BOS, 2, 5, 8] → 预测 EOS
    结果: [2, 5, 8] ✓
    """
    model.eval()
    src = src.to(device)
    
    # 编码器只需要运行一次（因为源序列不变）
    encoder_output = model.encode(src)
    
    # 初始化解码器输入：只有 BOS token
    decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    
    output_tokens = []
    
    for step in range(max_len):
        # 创建因果掩码
        tgt_mask = MiniTransformer.generate_causal_mask(
            decoder_input.size(1), device
        )
        
        # 解码器前向传播
        decoder_output = model.decode(decoder_input, encoder_output, tgt_mask=tgt_mask)
        
        # 只取最后一个位置的输出（因为我们要预测的是下一个 token）
        logits = model.output_projection(decoder_output[:, -1, :])  # [1, vocab_size]
        
        # 贪心：选概率最大的
        next_token = logits.argmax(dim=-1)  # [1]
        
        # 如果预测到 EOS，结束
        if next_token.item() == eos_token_id:
            break
        
        output_tokens.append(next_token.item())
        
        # 将新 token 拼接到解码器输入
        decoder_input = torch.cat([
            decoder_input,
            next_token.unsqueeze(0)  # [1, 1]
        ], dim=1)
    
    return output_tokens


# ============================================================
# 2. Beam Search 解码
# ============================================================
@torch.no_grad()
def beam_search_decode(
    model: MiniTransformer,
    src: torch.Tensor,         # [1, Ls]
    max_len: int,
    beam_size: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> list:
    """
    Beam Search 解码：维护 beam_size 个最优候选序列。
    
    核心思想：
    - 不是每步只保留一个最佳 token（贪心），而是保留 beam_size 个最佳序列
    - 在最终结果中选择总概率最高的序列
    
    为什么 Beam Search 更好？
    - 贪心可能在某一步选了局部最优但全局不优的 token
    - Beam Search 相当于在候选空间中做更广泛的搜索
    
    但注意：beam_size 越大不一定越好（可能导致生成过于保守/重复）
    """
    model.eval()
    src = src.to(device)
    
    encoder_output = model.encode(src)
    
    # 初始化 beam
    # 每个 beam 是 (序列, 累积 log 概率)
    beams = [(torch.tensor([[bos_token_id]], device=device), 0.0)]
    
    completed = []  # 已完成的序列
    
    for step in range(max_len):
        all_candidates = []
        
        for seq, score in beams:
            # 如果该 beam 已经生成了 EOS，直接保留
            if seq[0, -1].item() == eos_token_id:
                completed.append((seq, score))
                continue
            
            # 解码
            tgt_mask = MiniTransformer.generate_causal_mask(seq.size(1), device)
            
            # 扩展 encoder_output 以匹配当前序列
            decoder_output = model.decode(seq, encoder_output, tgt_mask=tgt_mask)
            logits = model.output_projection(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
            
            # 取 Top-K 个候选
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
            
            for i in range(beam_size):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))
        
        if not all_candidates:
            break
        
        # 保留 top beam_size 个候选
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]
    
    # 将未完成的也加入
    completed.extend(beams)
    
    if not completed:
        return []
    
    # 选择得分最高的序列
    best_seq, _ = max(completed, key=lambda x: x[1])
    
    # 去掉 BOS 和 EOS
    tokens = best_seq[0].tolist()
    if tokens[0] == bos_token_id:
        tokens = tokens[1:]
    if tokens and tokens[-1] == eos_token_id:
        tokens = tokens[:-1]
    
    return tokens


# ============================================================
# 3. 主推理流程
# ============================================================
def main():
    device = get_device()
    
    print("=" * 60)
    print("v01 Transformer 基础 - 序列排序推理")
    print("=" * 60)
    
    # 加载模型
    model = MiniTransformer(config).to(device)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        info = load_checkpoint(model, checkpoint_path, device=str(device))
        print(f"已加载模型 (epoch={info.get('epoch', '?')}, "
              f"val_loss={info.get('val_loss', '?'):.4f})")
    else:
        print("⚠️  未找到训练好的模型，将使用随机初始化的模型（结果会是随机的）")
        print("   请先运行 `python train.py` 进行训练\n")
    
    # 测试样例
    test_cases = [
        [5, 2, 8, 1, 3],
        [9, 7, 3, 6, 1, 4, 8],
        [15, 3, 22, 7, 11],
        [1, 1, 1, 2, 2],       # 有重复元素
        [29, 28, 27, 26, 25],   # 逆序
    ]
    
    print("\n" + "-" * 60)
    print("贪心解码 (Greedy Decoding)")
    print("-" * 60)
    
    for src_list in test_cases:
        src = torch.tensor([src_list], dtype=torch.long)
        result = greedy_decode(
            model, src,
            max_len=len(src_list) + 2,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            device=device,
        )
        expected = sorted(src_list)
        status = "✅" if result == expected else "❌"
        print(f"  输入: {src_list}")
        print(f"  预测: {result}")
        print(f"  正确: {expected} {status}")
        print()
    
    print("-" * 60)
    print(f"Beam Search 解码 (beam_size={config.beam_size})")
    print("-" * 60)
    
    for src_list in test_cases:
        src = torch.tensor([src_list], dtype=torch.long)
        result = beam_search_decode(
            model, src,
            max_len=len(src_list) + 2,
            beam_size=config.beam_size,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            device=device,
        )
        expected = sorted(src_list)
        status = "✅" if result == expected else "❌"
        print(f"  输入: {src_list}")
        print(f"  预测: {result}")
        print(f"  正确: {expected} {status}")
        print()


if __name__ == "__main__":
    main()
