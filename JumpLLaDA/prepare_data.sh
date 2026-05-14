#!/bin/bash
# ============================================================
# JumpLLaDA: One-Click Data Preparation
# ============================================================
# Downloads and tokenizes a dataset for JumpLLaDA training.
#
# Usage:
#   bash prepare_data.sh                    # Default: C4, 30B tokens
#   bash prepare_data.sh fineweb-edu 10B    # FineWeb-Edu, 10B tokens
#   bash prepare_data.sh slimpajama 30B     # SlimPajama, 30B tokens
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Arguments
DATASET="${1:-c4}"
NUM_TOKENS="${2:-30B}"
TOKENIZER="${3:-EleutherAI/gpt-neox-20b}"

echo "============================================================"
echo " JumpLLaDA Data Preparation"
echo "============================================================"
echo " Dataset:    $DATASET"
echo " Tokens:     $NUM_TOKENS"
echo " Tokenizer:  $TOKENIZER"
echo "============================================================"

# Install dependencies if needed
echo ""
echo "[1/2] Checking dependencies..."
pip install -q transformers datasets tqdm numpy 2>/dev/null || true

# Run data preparation
echo ""
echo "[2/2] Downloading and tokenizing dataset..."
python prepare_data.py \
    --dataset "$DATASET" \
    --num_tokens "$NUM_TOKENS" \
    --tokenizer "$TOKENIZER" \
    --seq_len 2048 \
    --num_proc 8

echo ""
echo "============================================================"
echo " Data preparation complete!"
echo " Next: bash run_train.sh"
echo "============================================================"
