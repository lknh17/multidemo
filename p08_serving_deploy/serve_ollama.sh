#!/bin/bash
# ============================================================
# p08 推理部署 - Ollama 本地部署脚本
#
# 自动创建 Modelfile，导入 GGUF 模型，并启动 Ollama 服务。
# 适合边缘设备、本地开发、离线场景。
#
# 使用方式:
#   bash serve_ollama.sh
#   bash serve_ollama.sh --gguf models/custom.gguf --tag my-model
# ============================================================

set -e

# ---- 默认参数 ----
GGUF_PATH="${GGUF_PATH:-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
MODEL_TAG="${MODEL_TAG:-qwen2.5-0.5b-custom}"
NUM_CTX=2048
NUM_GPU=99
NUM_THREAD=8
REPEAT_PENALTY=1.1
SYSTEM_PROMPT="你是一个有帮助的AI助手。请用中文回答问题。"
PORT=11434

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --gguf)     GGUF_PATH="$2"; shift 2 ;;
        --tag)      MODEL_TAG="$2"; shift 2 ;;
        --ctx)      NUM_CTX="$2"; shift 2 ;;
        --gpu)      NUM_GPU="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --system)   SYSTEM_PROMPT="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash serve_ollama.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --gguf PATH     GGUF 模型路径 (默认: $GGUF_PATH)"
            echo "  --tag NAME      模型标签 (默认: $MODEL_TAG)"
            echo "  --ctx SIZE      上下文大小 (默认: $NUM_CTX)"
            echo "  --gpu LAYERS    GPU 层数 (默认: $NUM_GPU, 99=全部)"
            echo "  --port PORT     服务端口 (默认: $PORT)"
            echo "  --system TEXT   系统提示词"
            echo "  -h, --help      显示帮助"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Ollama 本地部署"
echo "============================================================"
echo ""

# ---- Step 1: 检查 Ollama 是否安装 ----
echo "📋 Step 1: 检查 Ollama 安装..."
if ! command -v ollama &> /dev/null; then
    echo "  ✗ Ollama 未安装！"
    echo ""
    echo "  安装方式:"
    echo "    macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "    Windows:     下载 https://ollama.com/download"
    echo ""
    exit 1
fi
echo "  ✓ Ollama 版本: $(ollama --version 2>/dev/null || echo '未知')"
echo ""

# ---- Step 2: 检查 GGUF 模型文件 ----
echo "📋 Step 2: 检查模型文件..."
if [[ ! -f "$GGUF_PATH" ]]; then
    echo "  ✗ GGUF 模型文件不存在: $GGUF_PATH"
    echo ""
    echo "  获取 GGUF 模型的方法:"
    echo "    1. 从 HuggingFace 下载:"
    echo "       huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \\"
    echo "         qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/"
    echo ""
    echo "    2. 使用 llama.cpp 转换:"
    echo "       python convert_hf_to_gguf.py models/Qwen2.5-0.5B-Instruct \\"
    echo "         --outfile models/model.gguf --outtype q4_k_m"
    echo ""
    echo "  ⚠ 跳过 Ollama 部署，请先下载模型。"
    exit 1
fi

GGUF_SIZE=$(du -h "$GGUF_PATH" | cut -f1)
echo "  ✓ 模型文件: $GGUF_PATH ($GGUF_SIZE)"
echo ""

# ---- Step 3: 生成 Modelfile ----
echo "📋 Step 3: 生成 Modelfile..."
MODELFILE_PATH="Modelfile"

cat > "$MODELFILE_PATH" << EOF
# Ollama Modelfile - 自动生成
# 模型: $MODEL_TAG
# 时间: $(date '+%Y-%m-%d %H:%M:%S')

FROM $GGUF_PATH

# 推理参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty $REPEAT_PENALTY
PARAMETER num_ctx $NUM_CTX
PARAMETER num_gpu $NUM_GPU
PARAMETER num_thread $NUM_THREAD
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# 系统提示
SYSTEM """
$SYSTEM_PROMPT
"""

# 对话模板 (Qwen ChatML 格式)
TEMPLATE """
{{- if .System }}
<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}
<|im_start|>assistant
"""
EOF

echo "  ✓ Modelfile 已生成: $MODELFILE_PATH"
cat "$MODELFILE_PATH" | head -5
echo "  ..."
echo ""

# ---- Step 4: 创建 Ollama 模型 ----
echo "📋 Step 4: 创建 Ollama 模型 ($MODEL_TAG)..."
ollama create "$MODEL_TAG" -f "$MODELFILE_PATH"
echo "  ✓ 模型创建成功"
echo ""

# ---- Step 5: 验证模型 ----
echo "📋 Step 5: 验证模型..."
echo "  已安装的模型列表:"
ollama list
echo ""

# ---- Step 6: 快速测试 ----
echo "📋 Step 6: 快速测试..."
echo "  发送测试请求: '你好，请自我介绍。'"
echo ""
echo "--- 模型回复 ---"
echo "你好，请自我介绍。" | ollama run "$MODEL_TAG" --nowordwrap 2>/dev/null | head -20
echo ""
echo "--- 回复结束 ---"
echo ""

# ---- Step 7: 启动 API 服务 ----
echo "============================================================"
echo "  Ollama API 服务信息"
echo "============================================================"
echo "  模型标签:   $MODEL_TAG"
echo "  API 端口:   $PORT"
echo "  API 地址:   http://localhost:$PORT/api"
echo "  OpenAI 兼容: http://localhost:$PORT/v1"
echo ""
echo "  测试命令:"
echo "    curl http://localhost:$PORT/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$MODEL_TAG\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
echo ""
echo "  Ollama 服务通常已在后台运行。"
echo "  如需手动启动: OLLAMA_HOST=0.0.0.0:$PORT ollama serve"
echo "============================================================"
