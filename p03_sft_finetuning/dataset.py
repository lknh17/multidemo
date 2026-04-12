"""
p03 SFT 指令微调 - 数据集处理

实现多种 SFT 数据格式和 ChatML 模板：
1. Alpaca 格式 — instruction/input/output（最常见）
2. ShareGPT 格式 — conversations 多轮对话
3. ChatML 模板 — Qwen2.5 的对话模板
4. Label Masking — 仅对 assistant 回复计算 loss

核心要点：
- SFT 的关键区别于预训练：只对 assistant 的回复计算 loss
- 用户的问题部分 label 设为 -100，不参与 loss 计算
- 这样模型学会的是"如何回答"，而不是"如何提问"

使用方式:
    from dataset import create_sft_dataset
"""

import os
import sys
import json
import copy
from typing import Optional, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. ChatML 模板（Qwen2.5 对话格式）
# ============================================================
# Qwen2.5 使用 ChatML 格式：
# <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
# <|im_start|>user\n{问题}<|im_end|>\n
# <|im_start|>assistant\n{回答}<|im_end|>\n

CHATML_TEMPLATE = {
    "system_prefix": "<|im_start|>system\n",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "assistant_prefix": "<|im_start|>assistant\n",
    "assistant_suffix": "<|im_end|>\n",
    "default_system": "You are a helpful assistant.",
}


def build_chatml_prompt(
    instruction: str,
    input_text: str = "",
    output_text: str = "",
    system_message: str = None,
) -> str:
    """
    将 Alpaca 格式数据构建为 ChatML 对话格式。
    
    Args:
        instruction: 用户指令
        input_text: 额外输入（可选）
        output_text: 期望输出
        system_message: 系统提示（可选）
    
    Returns:
        完整的 ChatML 格式文本
    """
    tpl = CHATML_TEMPLATE
    system = system_message or tpl["default_system"]
    
    # 组合用户消息
    user_msg = instruction
    if input_text:
        user_msg = f"{instruction}\n\n{input_text}"
    
    # 构建完整对话
    text = (
        f"{tpl['system_prefix']}{system}{tpl['system_suffix']}"
        f"{tpl['user_prefix']}{user_msg}{tpl['user_suffix']}"
        f"{tpl['assistant_prefix']}{output_text}{tpl['assistant_suffix']}"
    )
    
    return text


def build_chatml_from_conversations(
    conversations: List[Dict[str, str]],
    system_message: str = None,
) -> str:
    """
    将 ShareGPT 多轮对话格式构建为 ChatML 格式。
    
    Args:
        conversations: [{"from": "human/gpt", "value": "..."}] 列表
        system_message: 系统提示
    
    Returns:
        完整的 ChatML 格式文本
    """
    tpl = CHATML_TEMPLATE
    system = system_message or tpl["default_system"]
    
    text = f"{tpl['system_prefix']}{system}{tpl['system_suffix']}"
    
    for conv in conversations:
        role = conv.get("from", conv.get("role", ""))
        content = conv.get("value", conv.get("content", ""))
        
        if role in ("human", "user"):
            text += f"{tpl['user_prefix']}{content}{tpl['user_suffix']}"
        elif role in ("gpt", "assistant"):
            text += f"{tpl['assistant_prefix']}{content}{tpl['assistant_suffix']}"
    
    return text


# ============================================================
# 2. Label Masking（核心：只对 assistant 回复计算 loss）
# ============================================================
def tokenize_with_label_mask(
    text: str,
    tokenizer,
    max_seq_length: int = 512,
) -> dict:
    """
    Tokenize 并生成 label mask。
    
    核心逻辑：
    1. 找到 assistant 回复的 token 范围
    2. 非 assistant 部分的 label 设为 -100（不计算 loss）
    3. assistant 部分的 label 保留原始 token_id（计算 loss）
    
    为什么要 label masking？
    - SFT 的目标是让模型学会回答，不是学会提问
    - 如果对 user 部分也计算 loss，模型会学习模仿用户提问的模式
    - 只对 assistant 计算 loss，模型专注学习如何给出好的回答
    
    Args:
        text: ChatML 格式的完整对话文本
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
    
    Returns:
        {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}
    """
    import torch
    
    tpl = CHATML_TEMPLATE
    
    # 编码完整文本
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = encoded["input_ids"].squeeze(0)        # [seq_len]
    attention_mask = encoded["attention_mask"].squeeze(0)  # [seq_len]
    
    # 默认全部 mask 掉（-100 表示不计算 loss）
    labels = torch.full_like(input_ids, -100)
    
    # 找到 assistant 回复的 token 范围
    # 策略：找 assistant_prefix 和 assistant_suffix 的 token 序列
    assistant_prefix_ids = tokenizer.encode(
        tpl["assistant_prefix"], add_special_tokens=False
    )
    assistant_suffix_ids = tokenizer.encode(
        tpl["assistant_suffix"], add_special_tokens=False
    )
    
    input_ids_list = input_ids.tolist()
    
    # 在 input_ids 中搜索 assistant 回复的位置
    i = 0
    while i < len(input_ids_list):
        # 查找 assistant_prefix 的起始位置
        prefix_found = False
        for j in range(i, len(input_ids_list) - len(assistant_prefix_ids) + 1):
            if input_ids_list[j:j+len(assistant_prefix_ids)] == assistant_prefix_ids:
                # 找到了 assistant 回复的开始
                start = j + len(assistant_prefix_ids)  # 跳过 prefix
                
                # 查找 assistant_suffix 的位置（回复结束）
                end = len(input_ids_list)
                for k in range(start, len(input_ids_list) - len(assistant_suffix_ids) + 1):
                    if input_ids_list[k:k+len(assistant_suffix_ids)] == assistant_suffix_ids:
                        end = k + len(assistant_suffix_ids)  # 包含 suffix
                        break
                
                # 将 assistant 回复部分的 label 设为原始 token_id
                labels[start:end] = input_ids[start:end]
                
                i = end
                prefix_found = True
                break
        
        if not prefix_found:
            break
    
    # pad 位置也设为 -100
    labels[attention_mask == 0] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 3. SFT 数据集类
# ============================================================
class SFTDataset:
    """
    SFT 指令微调数据集。
    
    支持两种输入格式：
    1. Alpaca: {"instruction": ..., "input": ..., "output": ...}
    2. ShareGPT: {"conversations": [{"from": "human/gpt", "value": ...}]}
    
    所有格式都会转换为 ChatML 模板，并应用 label masking。
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        data_format: str = "alpaca",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_seq_length: 最大序列长度
            data_format: 数据格式 (alpaca / sharegpt)
            max_samples: 最大样本数
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_format = data_format
        
        # 加载数据
        raw_data = self._load_data(data_path, max_samples)
        
        # 转换并 tokenize
        self.samples = self._process(raw_data)
    
    def _load_data(self, data_path: str, max_samples: Optional[int]) -> list:
        """加载 JSONL 数据"""
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if max_samples and len(samples) >= max_samples:
                        break
        print(f"  [SFT数据] 加载 {len(samples)} 条 ({self.data_format} 格式)")
        return samples
    
    def _process(self, raw_data: list) -> list:
        """处理数据：格式转换 + ChatML 构建 + Tokenize + Label Masking"""
        samples = []
        skipped = 0
        
        for item in raw_data:
            try:
                # 根据格式构建 ChatML 文本
                if self.data_format == "alpaca":
                    text = build_chatml_prompt(
                        instruction=item.get("instruction", ""),
                        input_text=item.get("input", ""),
                        output_text=item.get("output", ""),
                    )
                elif self.data_format == "sharegpt":
                    text = build_chatml_from_conversations(
                        conversations=item.get("conversations", []),
                    )
                else:
                    # 默认按 alpaca 处理
                    text = build_chatml_prompt(
                        instruction=item.get("instruction", ""),
                        input_text=item.get("input", ""),
                        output_text=item.get("output", ""),
                    )
                
                # Tokenize 并生成 label mask
                encoded = tokenize_with_label_mask(
                    text, self.tokenizer, self.max_seq_length
                )
                
                # 检查是否有有效的 label（至少有一些非 -100 的位置）
                valid_labels = (encoded["labels"] != -100).sum().item()
                if valid_labels < 5:
                    skipped += 1
                    continue
                
                samples.append(encoded)
                
            except Exception as e:
                skipped += 1
                continue
        
        print(f"  [SFT数据] 处理完成: {len(samples)} 条有效, {skipped} 条跳过")
        
        # 打印 label masking 统计
        if samples:
            import torch
            total_tokens = sum(s["attention_mask"].sum().item() for s in samples)
            loss_tokens = sum((s["labels"] != -100).sum().item() for s in samples)
            print(f"  [Label Mask] 总 token: {total_tokens:,}, "
                  f"计算 loss: {loss_tokens:,} ({loss_tokens/max(total_tokens,1)*100:.1f}%)")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 4. 统一创建函数
# ============================================================
def create_sft_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 512,
    data_format: str = "alpaca",
    max_samples: Optional[int] = None,
) -> SFTDataset:
    """
    创建 SFT 数据集。
    
    Args:
        data_path: JSONL 数据文件路径
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
        data_format: 数据格式 (alpaca / sharegpt)
        max_samples: 最大样本数
    
    Returns:
        SFTDataset 实例
    """
    print(f"\n[数据处理] SFT 数据集")
    print(f"  数据文件: {data_path}")
    print(f"  序列长度: {max_seq_length}")
    print(f"  数据格式: {data_format}")
    
    return SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        data_format=data_format,
        max_samples=max_samples,
    )
