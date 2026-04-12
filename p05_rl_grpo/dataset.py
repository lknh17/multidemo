"""
p05 强化学习 GRPO - 数据集处理

将 GSM8K 格式化为 Chain-of-Thought prompt，
支持答案提取（提取 #### 后的最终数值）。

使用方式:
    from dataset import create_grpo_dataset, extract_answer
"""

import os
import sys
import json
import re
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 数据加载
# ============================================================
def load_jsonl(file_path: str) -> list:
    """加载 JSONL 格式数据"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ============================================================
# 2. 答案提取
# ============================================================
def extract_answer(answer_text: str) -> Optional[float]:
    """
    从 GSM8K 答案中提取最终数值。
    
    GSM8K 格式：
        推理步骤...
        #### 最终数值
    
    Returns:
        提取的数值，如果无法提取返回 None
    """
    # 标准格式：#### 后跟数值
    if "####" in answer_text:
        ans_str = answer_text.split("####")[-1].strip()
        ans_str = ans_str.replace(",", "").replace("$", "")
        try:
            return float(ans_str)
        except ValueError:
            pass
    
    # 备选：尝试提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def extract_model_answer(response: str) -> Optional[float]:
    """
    从模型生成的响应中提取最终答案。
    
    模型可能用多种格式：
    - "答案是 42"
    - "最终答案：42"
    - "#### 42"
    - "The answer is 42"
    - "\\boxed{42}"
    """
    # 尝试 #### 格式
    if "####" in response:
        ans_str = response.split("####")[-1].strip()
        ans_str = ans_str.replace(",", "").replace("$", "")
        numbers = re.findall(r'-?\d+\.?\d*', ans_str)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
    
    # 尝试 \boxed{} 格式
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        ans_str = boxed[-1].replace(",", "")
        try:
            return float(ans_str)
        except ValueError:
            pass
    
    # 尝试 "答案是/答案：" 格式
    patterns = [
        r'答案[是为：:]\s*(-?\d+[\d,]*\.?\d*)',
        r'最终答案[是为：:]\s*(-?\d+[\d,]*\.?\d*)',
        r'[Tt]he answer is\s*(-?\d+[\d,]*\.?\d*)',
        r'[Aa]nswer:\s*(-?\d+[\d,]*\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
    
    # 最后尝试：取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


# ============================================================
# 3. Prompt 模板
# ============================================================
SYSTEM_PROMPT = """你是一个擅长数学推理的助手。请一步一步地解决数学问题。
要求：
1. 列出清晰的推理步骤
2. 每个步骤用"步骤 N："开头
3. 最后用"#### 最终答案"格式给出答案"""

COT_TEMPLATE = """请一步一步解决以下数学问题：

{question}

请按照以下格式回答：
步骤 1：...
步骤 2：...
...
#### 最终答案"""


def format_prompt(question: str, use_chat_template: bool = True) -> str:
    """将问题格式化为 CoT prompt"""
    if use_chat_template:
        return COT_TEMPLATE.format(question=question)
    return question


def format_chat_messages(question: str) -> list:
    """格式化为 chat 消息格式"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": COT_TEMPLATE.format(question=question)},
    ]


# ============================================================
# 4. GRPO 数据集
# ============================================================
class GRPODataset:
    """
    GRPO 数据集：为每个 prompt 准备用于 group 采样的格式。
    
    GRPO 的核心思路：
    1. 对每个 prompt 采样 G 个响应
    2. 用奖励函数评分
    3. 组内相对排名作为优势估计
    
    本数据集只准备 prompt 和 ground truth，
    采样和评分在训练循环中完成。
    """
    
    def __init__(
        self,
        samples: list,
        tokenizer=None,
        max_prompt_length: int = 256,
        use_chat_template: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data = self._process(samples, use_chat_template)
    
    def _process(self, samples: list, use_chat_template: bool) -> list:
        """处理数据"""
        processed = []
        
        for s in samples:
            question = s["question"]
            answer = s["answer"]
            ground_truth = extract_answer(answer)
            
            if ground_truth is None:
                continue  # 跳过无法提取答案的样本
            
            if use_chat_template and self.tokenizer:
                messages = format_chat_messages(question)
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = format_prompt(question, use_chat_template)
            
            processed.append({
                "prompt": prompt,
                "question": question,
                "ground_truth": ground_truth,
                "reference_answer": answer,
            })
        
        print(f"  [GRPO数据集] {len(samples)} 条 → {len(processed)} 条有效样本")
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 5. 统一创建函数
# ============================================================
def create_grpo_dataset(
    data_path: str,
    tokenizer=None,
    max_prompt_length: int = 256,
    max_samples: Optional[int] = None,
    use_chat_template: bool = True,
) -> GRPODataset:
    """
    创建 GRPO 数据集。
    
    Args:
        data_path: JSONL 数据文件路径
        tokenizer: HuggingFace tokenizer
        max_prompt_length: 最大 prompt 长度
        max_samples: 最大使用样本数
        use_chat_template: 是否使用 chat 模板
    """
    print(f"\n[数据处理] GRPO 数据集")
    print(f"  数据文件: {data_path}")
    
    samples = load_jsonl(data_path)
    
    if max_samples:
        samples = samples[:max_samples]
    print(f"  原始数据: {len(samples)} 条")
    
    return GRPODataset(
        samples=samples,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        use_chat_template=use_chat_template,
    )
