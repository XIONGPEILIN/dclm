#!/bin/bash
# ============================================================
# JumpLLaDA: One-Click Training Script
# ============================================================
# Trains the JumpLLaDA pure jump process language model.
#
# Prerequisites:
#   - Run `bash prepare_data.sh` first to prepare data
#   - Or specify --dataset_path manually
#
# Usage:
#   bash run_train.sh                           # Auto-detect GPUs
#   bash run_train.sh /path/to/tokenized_data   # Custom data path
#   CUDA_VISIBLE_DEVICES=0,1 bash run_train.sh  # Specific GPUs
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Configuration (edit these as needed)
# ============================================================
MAX_STEPS="${MAX_STEPS:-100000}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-3e-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/jump_llada}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
RESUME_FROM="${RESUME_FROM:-}"

# ============================================================
# Auto-detect data path
# ============================================================
DATA_PATH="${1:-}"

if [ -z "$DATA_PATH" ]; then
    # Try to find tokenized data
    if [ -d "./data/tokenized" ]; then
        # Find first subdirectory with .bin files
        DATA_PATH=$(find ./data/tokenized -name "*.bin" -print -quit 2>/dev/null | xargs -r dirname)
    fi
    if [ -z "$DATA_PATH" ] && [ -d "/export/space0/data/HF/jump_llada_data/tokenized" ]; then
        DATA_PATH=$(find /export/space0/data/HF/jump_llada_data/tokenized -name "*.bin" -print -quit 2>/dev/null | xargs -r dirname)
    fi
fi

if [ -z "$DATA_PATH" ] || [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: No tokenized data found."
    echo "Run 'bash prepare_data.sh' first, or specify path: bash run_train.sh /path/to/data"
    exit 1
fi

echo "============================================================"
echo " JumpLLaDA Training"
echo "============================================================"
echo " Data path:     $DATA_PATH"
echo " Max steps:     $MAX_STEPS"
echo " Per-device BS: $PER_DEVICE_BS"
echo " Grad accum:    $GRAD_ACCUM"
echo " Learning rate: $LR"
echo " Seq length:    $SEQ_LEN"
echo " Output dir:    $OUTPUT_DIR"

# ============================================================
# Detect GPU count
# ============================================================
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo " GPUs:          $NUM_GPUS"
else
    NUM_GPUS=0
    echo " GPUs:          None (CPU mode)"
fi
echo "============================================================"

# ============================================================
# Install dependencies
# ============================================================
echo ""
echo "[1/2] Checking dependencies..."
pip install -q torch transformers accelerate datasets numpy tqdm 2>/dev/null || true

# ============================================================
# Build training command
# ============================================================
TRAIN_ARGS="--dataset_path $DATA_PATH \
    --max_steps $MAX_STEPS \
    --per_device_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --seq_len $SEQ_LEN \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing"

if [ -n "$WANDB_PROJECT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --wandb_project $WANDB_PROJECT"
fi

if [ -n "$RESUME_FROM" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume_from $RESUME_FROM"
fi

# ============================================================
# Launch training
# ============================================================
echo ""
echo "[2/2] Starting training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching with accelerate (multi-GPU)..."
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes "$NUM_GPUS" \
        train.py $TRAIN_ARGS
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching single-GPU training..."
    python train.py $TRAIN_ARGS
else
    echo "Launching CPU training (slow!)..."
    python train.py $TRAIN_ARGS
fi

echo ""
echo "============================================================"
echo " Training complete!"
echo " Checkpoints saved to: $OUTPUT_DIR"
echo "============================================================"
