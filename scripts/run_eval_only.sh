#!/bin/bash
#SBATCH --job-name=twibot_eval
#SBATCH --chdir=/sciclone/home/hwhyman/Graph_learning
#SBATCH --output=logs/twibot_eval_%j.log
#SBATCH --error=logs/twibot_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# Suppress HuggingFace tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Setup
cd /sciclone/home/hwhyman/Graph_learning
source .venv/bin/activate
mkdir -p logs

EVAL_SPLIT=${EVAL_SPLIT:-test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_NEIGHBORS=${NUM_NEIGHBORS:-"15 10"}
CHECKPOINT=${CHECKPOINT:-""}
PROCESSED_DIR=${PROCESSED_DIR:-processed}
RAW_DATA=${RAW_DATA:-TwiBot-22}

ENABLE_TWEET_TEXT=${ENABLE_TWEET_TEXT:-true}
TWEET_TEXT_MODEL=${TWEET_TEXT_MODEL:-"xlm-roberta-base"}
TWEET_TEXT_MAX_LEN=${TWEET_TEXT_MAX_LEN:-128}
TWEET_TEXT_BATCH=${TWEET_TEXT_BATCH:-512}

echo "=============================================="
echo "TwiBot-22 Evaluation"
echo "=============================================="
echo "Split: $EVAL_SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "Checkpoint: ${CHECKPOINT:-'(auto: processed/best_model.pt)'}"
echo ""
echo "Note: Model architecture will be auto-detected from checkpoint"
echo "=============================================="

# Build optional args
CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi

TWEET_TEXT_ARGS=""
if [ "$ENABLE_TWEET_TEXT" = true ]; then
    TWEET_TEXT_ARGS="--enable-tweet-text --tweet-text-model $TWEET_TEXT_MODEL --tweet-text-max-length $TWEET_TEXT_MAX_LEN --tweet-text-batch-size $TWEET_TEXT_BATCH"
fi

python eval.py \
    --raw-data "$RAW_DATA" \
    --processed-dir "$PROCESSED_DIR" \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --num-neighbors $NUM_NEIGHBORS \
    --split "$EVAL_SPLIT" \
    --use-hgt \
    $CHECKPOINT_ARG \
    $TWEET_TEXT_ARGS

echo ""
echo "==> Evaluation complete!"
