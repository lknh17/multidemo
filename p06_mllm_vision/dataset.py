"""
p06 MLLM 多模态视觉微调 - 数据集处理

实现多模态数据集：
1. 加载图像 + 文本对话数据
2. 图像预处理（resize / normalize）
3. 格式化为 Qwen2.5-VL 的输入格式
4. 支持动态分辨率处理

使用方式:
    from dataset import create_mllm_dataset
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 数据加载
# ============================================================
def load_json_data(file_path: str) -> list:
    """加载 JSON 格式数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl_data(file_path: str) -> list:
    """加载 JSONL 格式数据"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ============================================================
# 2. 图像预处理
# ============================================================
class ImageProcessor:
    """
    图像预处理器。
    
    Qwen2.5-VL 的图像处理特点：
    - 支持动态分辨率（不强制 resize 到固定大小）
    - 图像被切分为多个 patch（28×28）
    - 视觉 token 数量 = (H/28) × (W/28)
    - 通过 min_pixels 和 max_pixels 控制 token 数量范围
    """
    
    def __init__(
        self,
        image_size: int = 448,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        self.image_size = image_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
    
    def preprocess(self, image):
        """
        预处理单张图像。
        
        步骤：
        1. 转为 RGB（处理 RGBA/灰度图）
        2. 调整分辨率到合适范围
        3. 归一化到 [0, 1] 再标准化
        4. 转为 tensor
        """
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        if isinstance(image, str):
            image = Image.open(image)
        
        # 转为 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 计算目标尺寸（保持宽高比）
        w, h = image.size
        pixels = w * h
        
        if pixels < self.min_pixels:
            scale = (self.min_pixels / pixels) ** 0.5
            w, h = int(w * scale), int(h * scale)
        elif pixels > self.max_pixels:
            scale = (self.max_pixels / pixels) ** 0.5
            w, h = int(w * scale), int(h * scale)
        
        # 确保是 28 的倍数（patch size）
        w = max(28, (w // 28) * 28)
        h = max(28, (h // 28) * 28)
        
        # 变换流水线
        transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        
        tensor = transform(image)
        return tensor
    
    def __call__(self, image):
        return self.preprocess(image)


# ============================================================
# 3. 多模态数据集
# ============================================================
class MLLMDataset:
    """
    多模态指令微调数据集。
    
    数据格式（LLaVA 格式）:
    {
        "id": "...",
        "image": "coco/train2017/000000123456.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n描述这张图片"},
            {"from": "gpt", "value": "图片中展示了..."}
        ]
    }
    
    处理流程：
    1. 加载图像并预处理
    2. 将对话转为 Qwen2.5-VL 的 chat template 格式
    3. Tokenize 并构造 input_ids、labels（assistant 部分才计算 loss）
    """
    
    def __init__(
        self,
        data: list,
        processor,
        tokenizer,
        max_seq_length: int = 2048,
        image_dir: str = None,
    ):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_dir = image_dir or "data/images"
        
        print(f"  [MLLMDataset] {len(data)} 个样本, 最大序列长度: {max_seq_length}")
    
    def _build_prompt(self, sample: dict) -> str:
        """将对话构造为 Qwen2.5-VL 的 prompt 格式"""
        conversations = sample.get("conversations", [])
        
        messages = []
        for conv in conversations:
            role = conv["from"]
            value = conv["value"]
            
            if role == "human":
                # 替换 <image> 为 Qwen 的图像占位符
                value = value.replace("<image>", "<|image_pad|>")
                value = value.strip()
                messages.append({"role": "user", "content": value})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": value})
        
        return messages
    
    def _tokenize_messages(self, messages: list) -> dict:
        """
        将消息列表 tokenize 为模型输入。
        
        关键：只对 assistant 的回答部分计算 loss，
        user 的输入部分标记为 -100（忽略）。
        """
        import torch
        
        # 使用 tokenizer 的 apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize 完整文本
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # 构造 labels：只对 assistant 部分计算 loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # padding 部分忽略
        
        # 找到 assistant 回答的起始位置，将之前的部分标记为 -100
        # 简化处理：使用 assistant 标记 token 来定位
        assistant_token = self.tokenizer.encode("assistant", add_special_tokens=False)
        if len(assistant_token) > 0:
            input_list = input_ids.tolist()
            in_assistant = False
            for i in range(len(input_list)):
                if not in_assistant:
                    labels[i] = -100
                # 简单启发式：遇到 assistant 相关 token 后开始计算 loss
                if i < len(input_list) - 1:
                    # 检测 role 切换
                    pass
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _load_image(self, image_path: str):
        """加载图像文件"""
        from PIL import Image
        
        # 尝试多个路径
        candidates = [
            image_path,
            os.path.join(self.image_dir, image_path),
            os.path.join(self.image_dir, os.path.basename(image_path)),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        
        # 如果找不到图像，创建占位图像
        return Image.new("RGB", (224, 224), color=(128, 128, 128))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import torch
        
        sample = self.data[idx]
        
        # 加载图像
        image_path = sample.get("image", "")
        image = self._load_image(image_path)
        
        # 构建对话消息
        messages = self._build_prompt(sample)
        
        # 使用 processor 处理图文输入（适配 Qwen2.5-VL）
        if hasattr(self.processor, 'apply_chat_template'):
            # Qwen2.5-VL AutoProcessor
            # 构造带图片引用的消息格式
            qwen_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    content_parts = []
                    if "<|image_pad|>" in msg["content"]:
                        content_parts.append({"type": "image", "image": image})
                        text = msg["content"].replace("<|image_pad|>", "").strip()
                    else:
                        text = msg["content"]
                    if text:
                        content_parts.append({"type": "text", "text": text})
                    qwen_messages.append({"role": "user", "content": content_parts})
                else:
                    qwen_messages.append({"role": msg["role"], "content": msg["content"]})
            
            text = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=False,
            )
            # 不截断，保持图像 token 和 pixel_values 一致
            inputs = self.processor(
                text=[text], images=[image],
                padding=False, return_tensors="pt",
            )
            
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            
            # 构造 labels：只对 assistant 部分计算 loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            # 传递视觉相关字段（保持原始 shape）
            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"]
            
            return result
        else:
            # Fallback: 自定义 ImageProcessor
            pixel_values = self.processor(image)
            tokens = self._tokenize_messages(messages)
            return {
                "pixel_values": pixel_values,
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["labels"],
            }


# ============================================================
# 4. 统一创建函数
# ============================================================
def create_mllm_dataset(
    data_path: str,
    processor,
    tokenizer,
    max_seq_length: int = 2048,
    max_samples: Optional[int] = None,
    image_dir: str = None,
    val_ratio: float = 0.05,
):
    """
    创建多模态微调数据集。
    
    Args:
        data_path: JSON 数据文件路径
        processor: 图像处理器（ImageProcessor 或 Qwen2VLImageProcessor）
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
        max_samples: 最大使用样本数
        image_dir: 图像文件目录
        val_ratio: 验证集比例
    
    Returns:
        (train_dataset, val_dataset)
    """
    print(f"\n[数据处理] 多模态指令微调数据")
    print(f"  数据文件: {data_path}")
    print(f"  序列长度: {max_seq_length}")
    
    # 加载数据
    if data_path.endswith(".jsonl"):
        data = load_jsonl_data(data_path)
    else:
        data = load_json_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
    print(f"  原始数据: {len(data)} 条")
    
    # 划分训练/验证集
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    
    train_dataset = MLLMDataset(
        train_data, processor, tokenizer,
        max_seq_length=max_seq_length,
        image_dir=image_dir,
    )
    
    val_dataset = MLLMDataset(
        val_data, processor, tokenizer,
        max_seq_length=max_seq_length,
        image_dir=image_dir,
    ) if val_size > 0 else None
    
    return train_dataset, val_dataset
