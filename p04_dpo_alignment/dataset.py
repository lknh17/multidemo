"""
p04 DPO 对齐训练 - 数据集处理

实现偏好数据的加载与预处理：
1. 加载 prompt / chosen / rejected 三元组
2. 应用 chat template 格式化
3. Tokenize 并截断到指定长度

使用方式:
    from dataset import create_preference_dataset
"""

import os
import sys
import json
from typing import Optional, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 数据加载
# ============================================================
def load_preference_data(file_path: str) -> List[Dict]:
    """
    加载 JSONL 格式的偏好数据。
    
    每条数据格式：
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # 验证必要字段
                if all(k in item for k in ("prompt", "chosen", "rejected")):
                    samples.append(item)
    
    print(f"  加载 {len(samples)} 条偏好数据对")
    return samples


# ============================================================
# 2. Chat Template 格式化
# ============================================================
def format_as_chat(prompt: str, response: str, tokenizer) -> str:
    """
    将 prompt + response 格式化为 chat template。
    
    大多数 Instruct 模型使用特定的 chat template，例如：
    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    
    使用 tokenizer.apply_chat_template 自动处理不同模型的格式差异。
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    
    # 尝试使用模型自带的 chat template
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return formatted
    except Exception:
        # fallback：简单拼接
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


# ============================================================
# 3. 偏好数据集
# ============================================================
class PreferenceDataset:
    """
    偏好数据集：加载 prompt/chosen/rejected 三元组。
    
    DPO 训练需要的数据格式：
    - prompt: 用户输入
    - chosen: 人类/GPT-4 偏好的回复
    - rejected: 被拒绝的回复
    
    DPOTrainer 要求返回的字段：
    - prompt: str
    - chosen: str（完整的 prompt + chosen response，chat template 格式）
    - rejected: str（完整的 prompt + rejected response，chat template 格式）
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.samples = self._process(data, tokenizer)
    
    def _process(self, data: List[Dict], tokenizer) -> List[Dict]:
        """处理偏好数据为 DPOTrainer 需要的格式"""
        samples = []
        skipped = 0
        
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            # 格式化为 chat template
            chosen_formatted = format_as_chat(prompt, chosen, tokenizer)
            rejected_formatted = format_as_chat(prompt, rejected, tokenizer)
            
            # 检查长度（粗略估计）
            chosen_tokens = tokenizer.encode(chosen_formatted, add_special_tokens=False)
            rejected_tokens = tokenizer.encode(rejected_formatted, add_special_tokens=False)
            
            if len(chosen_tokens) > self.max_length or len(rejected_tokens) > self.max_length:
                skipped += 1
                continue
            
            samples.append({
                "prompt": prompt,
                "chosen": chosen_formatted,
                "rejected": rejected_formatted,
            })
        
        if skipped > 0:
            print(f"  跳过 {skipped} 条超长样本")
        print(f"  [偏好数据] {len(data)} 条原始 → {len(samples)} 条有效样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 4. KTO 数据集（不需要配对）
# ============================================================
class KTODataset:
    """
    KTO（Kahneman-Tversky Optimization）数据集。
    
    KTO 的特点是不需要 chosen/rejected 配对，
    只需要标注每个回复是"好"还是"坏"。
    
    数据格式：
    - prompt: str
    - completion: str（回复文本）
    - label: bool（True=好回复，False=坏回复）
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._process(data, tokenizer)
    
    def _process(self, data: List[Dict], tokenizer) -> List[Dict]:
        """将配对数据转换为 KTO 格式"""
        samples = []
        
        for item in data:
            prompt = item["prompt"]
            
            # chosen → label=True
            chosen_formatted = format_as_chat(prompt, item["chosen"], tokenizer)
            samples.append({
                "prompt": prompt,
                "completion": chosen_formatted,
                "label": True,
            })
            
            # rejected → label=False
            rejected_formatted = format_as_chat(prompt, item["rejected"], tokenizer)
            samples.append({
                "prompt": prompt,
                "completion": rejected_formatted,
                "label": False,
            })
        
        print(f"  [KTO数据] {len(data)} 条配对 → {len(samples)} 条单条样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 5. 统一创建函数
# ============================================================
def create_preference_dataset(
    data_path: str,
    tokenizer,
    algorithm: str = "dpo",
    max_length: int = 1024,
    max_prompt_length: int = 512,
    max_samples: Optional[int] = None,
):
    """
    创建偏好数据集。
    
    Args:
        data_path: JSONL 数据文件路径
        tokenizer: HuggingFace tokenizer
        algorithm: 算法类型 (dpo/simpo/orpo/kto)
        max_length: 最大序列长度
        max_prompt_length: prompt 最大长度
        max_samples: 最大使用样本数
    
    Returns:
        PreferenceDataset 或 KTODataset
    """
    print(f"\n[偏好数据处理] 算法: {algorithm}")
    print(f"  数据文件: {data_path}")
    print(f"  序列长度: {max_length} (prompt: {max_prompt_length})")
    
    data = load_preference_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
    print(f"  使用数据: {len(data)} 条")
    
    if algorithm == "kto":
        return KTODataset(data, tokenizer, max_length)
    else:
        return PreferenceDataset(data, tokenizer, max_length, max_prompt_length)
