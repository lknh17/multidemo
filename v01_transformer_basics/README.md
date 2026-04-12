# v01: Transformer 基础

## 任务
用 Mini Transformer (Encoder-Decoder) 学习数字排序：输入 `[5,2,8,1,3]` → 输出 `[1,2,3,5,8]`

## 你将学到
- 自注意力机制 (Scaled Dot-Product Attention)
- 多头注意力 (Multi-Head Attention)
- 正弦位置编码
- 前馈网络 (FFN) 与 GELU 激活
- 残差连接 + Pre-Norm (LayerNorm)
- 完整的 Encoder-Decoder 架构
- Teacher Forcing 训练 + 贪心/Beam Search 推理

## 快速开始

```bash
pip install -r requirements.txt
python demo_data/generate_data.py   # 生成 Demo 数据
python train.py                      # 训练（约 2-5 分钟）
python inference.py                  # 推理测试
```

## 预期输出
- 训练 50 个 epoch 后，序列准确率应达到 **95%+**
- 推理时大部分排序结果应完全正确
