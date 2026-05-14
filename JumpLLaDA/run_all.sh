#!/bin/bash
# ============================================================
# JumpLLaDA: 一键数据准备 + 训练脚本
# ============================================================
# 适配: 8× A40-48GB
#
# 用法:
#   cd ~/dclm/JumpLLaDA
#   bash run_all.sh
# ============================================================

set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID                                                                                                                                                 
export NCCL_P2P_DISABLE=0                                                                                                                                                           
export NCCL_SHM_DISABLE=1                                                                                                                                                           
export NCCL_P2P_LEVEL=5 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " JumpLLaDA: 数据准备 + 训练"
echo "============================================================"

# ============================================================
# 第1步: 安装依赖
# ============================================================
echo ""
echo "[1/3] 安装依赖..."
uv pip install  torch transformers accelerate datasets numpy tqdm wandb 2>/dev/null || true

# ============================================================
# 第2步已移除：使用边下边练的 Streaming 模式
# ============================================================
echo ""
echo "[2/3] 数据: 直接从 HuggingFace 边下载边分词 (Streaming) ..."

# ============================================================
# 第3步: 训练 (8× A40-48GB)
# ============================================================
echo ""
echo "[3/3] 启动训练..."

# --- 训练参数 (根据 8× A40-48GB 配置) ---
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")
PER_DEVICE_BS=2               # 每卡 batch size (48GB 显存安全值)
GRAD_ACCUM=8                  # 梯度累积步数
# 有效 batch size = NUM_GPUS × PER_DEVICE_BS × GRAD_ACCUM
#                 = 8 × 2 × 8 = 128
LR=3e-4
MAX_STEPS=100000
SEQ_LEN=2048
OUTPUT_DIR="/export/ssd2/xiong-p/dclm/outputs/jump_llada"

# 创建输出目录软链接
if [ ! -L "./outputs" ]; then
    if [ -d "./outputs" ]; then
        echo "警告: ./outputs 已存在且不是软链接，跳过软链接创建"
    else
        mkdir -p /export/ssd2/xiong-p/dclm/outputs
        ln -s /export/ssd2/xiong-p/dclm/outputs ./outputs
        echo "已创建软链接: ./outputs -> /export/ssd2/xiong-p/dclm/outputs"
    fi
fi

# 清理可能存在的随机测试数据，防止干扰全量训练
if [ -d "/export/ssd2/xiong-p/dclm/data/tokenized/test" ]; then
    echo "清理随机测试数据: /export/ssd2/xiong-p/dclm/data/tokenized/test"
    rm -rf /export/ssd2/xiong-p/dclm/data/tokenized/test
fi

echo "  GPU 数量:       $NUM_GPUS"
echo "  每卡 BS:        $PER_DEVICE_BS"
echo "  梯度累积:       $GRAD_ACCUM"
echo "  有效 BS:        $((NUM_GPUS * PER_DEVICE_BS * GRAD_ACCUM))"
echo "  学习率:         $LR"
echo "  最大步数:       $MAX_STEPS"
echo "  序列长度:       $SEQ_LEN"
echo "  输出目录:       $OUTPUT_DIR"
echo "============================================================"

TRAIN_ARGS="--dataset_name allenai/c4 \
    --dataset_subset en \
    --streaming \
    --max_steps $MAX_STEPS \
    --per_device_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --seq_len $SEQ_LEN \
    --output_dir $OUTPUT_DIR \
    --wandb_project jump_llada"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "使用 accelerate 多卡训练 ($NUM_GPUS GPUs)..."
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes "$NUM_GPUS" \
        train.py $TRAIN_ARGS
else
    echo "单卡训练..."
    python train.py $TRAIN_ARGS
fi

echo ""
echo "============================================================"
echo " 训练完成! 模型保存在: $OUTPUT_DIR"
echo "============================================================"
