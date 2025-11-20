#!/bin/bash
#SBATCH --job-name=twibot_train_eval
#SBATCH --chdir=/sciclone/home/hwhyman/Graph_learning
#SBATCH --output=logs/twibot_train_eval_%j.log
#SBATCH --error=logs/twibot_train_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# Setup
cd /sciclone/home/hwhyman/Graph_learning
source .venv/bin/activate
mkdir -p logs

# ============ EDIT THESE VALUES ============
HIDDEN_DIM=256
BATCH_SIZE=64
LR_BACKBONE=1.5e-4
MAX_GRAD_NORM=1.0
EVAL_SPLIT=test
TEXT_MODEL="xlm-roberta-base"
TWEET_TEXT_MODEL=""  # Leave empty to reuse TEXT_MODEL
TWEET_TEXT_MAX_LEN=96
TWEET_TEXT_BATCH=512
# Set TWEET_TEXT_ARGS="" below to disable tweet-node text features entirely.
REBUILD_FLAG=""      # Set to "--rebuild" to force preprocessing rebuild
# ===========================================
echo "==> Environment variables:"
echo "HIDDEN_DIM=$HIDDEN_DIM"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LR_BACKBONE=$LR_BACKBONE"
echo "MAX_GRAD_NORM=$MAX_GRAD_NORM"
echo "EVAL_SPLIT=$EVAL_SPLIT"
echo "TEXT_MODEL=$TEXT_MODEL"
SELECTED_TWEET_TEXT_MODEL=${TWEET_TEXT_MODEL:-$TEXT_MODEL}
TWEET_TEXT_ARGS="--enable-tweet-text --tweet-text-model $SELECTED_TWEET_TEXT_MODEL --tweet-text-max-length $TWEET_TEXT_MAX_LEN --tweet-text-batch-size $TWEET_TEXT_BATCH"
echo "TWEET_TEXT_ARGS=$TWEET_TEXT_ARGS"
echo "REBUILD_FLAG=$REBUILD_FLAG"

# Train
echo "==> Training model (hidden_dim=$HIDDEN_DIM)"
python train.py \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --lr-backbone $LR_BACKBONE \
    --max-grad-norm $MAX_GRAD_NORM \
    --text-model "$TEXT_MODEL" \
    --use-hgt \
    --use-lora \
    $TWEET_TEXT_ARGS \
    $REBUILD_FLAG \
    # --resume


# Evaluate
echo "==> Evaluating on $EVAL_SPLIT split"
python eval.py \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --split "$EVAL_SPLIT" \
    --text-model "$TEXT_MODEL" \
    --use-hgt \
    --use-lora \
    $TWEET_TEXT_ARGS

echo "==> Done!"
