"""
v14 RLHF / DPO 偏好对齐 - 偏好数据集

提供两种数据集：
1. PreferenceDataset — (prompt, chosen, rejected) 偏好三元组
2. KTODataset — (prompt, response, is_desirable) 二元标注
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PreferenceDataset(Dataset):
    """
    偏好对数据集，用于 DPO / SimPO / RLHF 训练。
    
    每条数据包含：
    - prompt: 用户输入
    - chosen_response: 人类偏好的回复
    - rejected_response: 人类不偏好的回复
    
    教学用途：使用合成数据（排序任务）：
    - chosen: 正确排序结果
    - rejected: 错误排序结果（引入随机扰动）
    """
    
    def __init__(self, num_samples: int = 2000, max_prompt_len: int = 32,
                 max_response_len: int = 64, vocab_size: int = 5000, seed: int = 42):
        self.num_samples = num_samples
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        
        # BOS/EOS/PAD token ids
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        """生成合成偏好数据"""
        data = []
        for _ in range(self.num_samples):
            # 生成随机 prompt
            prompt_len = self.rng.randint(8, self.max_prompt_len)
            prompt = self.rng.randint(3, self.vocab_size, size=prompt_len).tolist()
            
            # Chosen: 排序后的序列（正确答案）
            response_len = self.rng.randint(8, self.max_response_len)
            chosen_response = sorted(
                self.rng.randint(3, self.vocab_size, size=response_len).tolist()
            )
            
            # Rejected: 带扰动的排序（错误答案）
            rejected_response = chosen_response.copy()
            # 随机交换一些位置
            n_swaps = max(1, len(rejected_response) // 4)
            for _ in range(n_swaps):
                i, j = self.rng.randint(0, len(rejected_response), size=2)
                rejected_response[i], rejected_response[j] = \
                    rejected_response[j], rejected_response[i]
            
            data.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })
        return data
    
    def _build_sequence(self, prompt, response, max_len):
        """构建完整序列: [BOS] + prompt + response + [EOS] + [PAD...]"""
        seq = [self.bos_id] + prompt + response + [self.eos_id]
        
        # 构建 labels: prompt 部分为 -100 (不计算 loss)
        labels = [-100] * (1 + len(prompt)) + response + [self.eos_id]
        
        # Padding
        if len(seq) < max_len:
            pad_len = max_len - len(seq)
            seq = seq + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            seq = seq[:max_len]
            labels = labels[:max_len]
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        max_len = self.max_prompt_len + self.max_response_len + 2
        
        chosen_ids, chosen_labels = self._build_sequence(
            item["prompt"], item["chosen"], max_len
        )
        rejected_ids, rejected_labels = self._build_sequence(
            item["prompt"], item["rejected"], max_len
        )
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


class KTODataset(Dataset):
    """
    KTO 数据集 — 只需要 good/bad 标注，不需要偏好对。
    
    每条数据：(prompt, response, is_desirable)
    """
    
    def __init__(self, num_samples: int = 2000, max_prompt_len: int = 32,
                 max_response_len: int = 64, vocab_size: int = 5000,
                 desirable_ratio: float = 0.6, seed: int = 42):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        self.bos_id, self.eos_id, self.pad_id = 1, 2, 0
        self.max_len = max_prompt_len + max_response_len + 2
        
        self.data = []
        for _ in range(num_samples):
            prompt_len = self.rng.randint(8, max_prompt_len)
            prompt = self.rng.randint(3, vocab_size, size=prompt_len).tolist()
            
            response_len = self.rng.randint(8, max_response_len)
            is_desirable = self.rng.random() < desirable_ratio
            
            if is_desirable:
                response = sorted(self.rng.randint(3, vocab_size, size=response_len).tolist())
            else:
                response = self.rng.randint(3, vocab_size, size=response_len).tolist()
            
            self.data.append({
                "prompt": prompt,
                "response": response,
                "is_desirable": is_desirable,
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        seq = [self.bos_id] + item["prompt"] + item["response"] + [self.eos_id]
        labels = [-100] * (1 + len(item["prompt"])) + item["response"] + [self.eos_id]
        
        if len(seq) < self.max_len:
            seq += [self.pad_id] * (self.max_len - len(seq))
            labels += [-100] * (self.max_len - len(labels))
        else:
            seq = seq[:self.max_len]
            labels = labels[:self.max_len]
        
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_desirable": torch.tensor(item["is_desirable"], dtype=torch.bool),
        }


def create_preference_dataloader(num_samples=2000, batch_size=16, **kwargs):
    """创建偏好对数据加载器"""
    dataset = PreferenceDataset(num_samples=num_samples, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def create_kto_dataloader(num_samples=2000, batch_size=16, **kwargs):
    """创建 KTO 数据加载器"""
    dataset = KTODataset(num_samples=num_samples, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
