#!/bin/bash
# p06 三种冻结策略对比实验
cd "$(dirname "$0")"

# 清除 pyc 缓存
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find .. -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

SAMPLES=200

echo "============================================================"
echo "  p06 三种冻结策略对比实验 (${SAMPLES} samples, 1 epoch)"
echo "============================================================"

# 实验1: freeze_vision + LoRA
echo ""
echo ">>> [1/3] freeze_vision + LoRA"
python3 -u train.py \
    --freeze-strategy freeze_vision \
    --max-samples $SAMPLES \
    --output-dir outputs/exp1_freeze_vision_lora 2>&1 | tee outputs/exp1_freeze_vision_lora.log
echo ">>> [1/3] 完成"

# 实验2: partial_unfreeze + LoRA
echo ""
echo ">>> [2/3] partial_unfreeze + LoRA"
python3 -u train.py \
    --freeze-strategy partial_unfreeze \
    --unfreeze-layers 4 \
    --max-samples $SAMPLES \
    --output-dir outputs/exp2_partial_unfreeze_lora 2>&1 | tee outputs/exp2_partial_unfreeze_lora.log
echo ">>> [2/3] 完成"

# 实验3: full + no-lora
echo ""
echo ">>> [3/3] full + no-lora"
python3 -u train.py \
    --freeze-strategy full \
    --no-lora \
    --learning-rate 1e-5 \
    --max-samples $SAMPLES \
    --output-dir outputs/exp3_full_no_lora 2>&1 | tee outputs/exp3_full_no_lora.log
echo ">>> [3/3] 完成"

echo ""
echo "============================================================"
echo "  所有实验完成！"
echo "============================================================"
