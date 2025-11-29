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

# Suppress HuggingFace tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Setup
cd /sciclone/home/hwhyman/Graph_learning
source .venv/bin/activate
mkdir -p logs

# ============ CONFIGURATION ============
# Model Architecture
HIDDEN_DIM=256
NUM_LAYERS=3
GRAPH_HEADS=4
DROPOUT=0.4                # Increased from 0.3 to reduce overfitting

# Training Hyperparameters
BATCH_SIZE=128              # Reduced for GPU memory (LoRA + trainable text encoder needs more VRAM)
EPOCHS=15
PATIENCE=5
LR_BACKBONE=1e-4           # Reduced from 1.5e-4 for stability
LR_TEXT=2e-5               # Lower LR for text encoder
WEIGHT_DECAY=0.05          # Increased from 0.01 to reduce overfitting
MAX_GRAD_NORM=1.0

# Class Imbalance - use weight multiplier only (not both oversampling AND weighting)
BOT_WEIGHT_MULT=2.0        # Moderate class weight (3.0 was too aggressive, causing overconfidence)
TARGET_POS_RATIO=""        # Disabled - using class weights instead
LABEL_SMOOTHING=0.1        # Helps with overfitting and calibration

# Graph Sampling
NUM_NEIGHBORS="20 10"    # Reduced from "25 15" for faster training

# Text Encoder
TEXT_MODEL="xlm-roberta-base"
TEXT_MAX_LEN=256           # Reduced from 384 to save GPU memory
TEXT_TRAINABLE=""        # Set to "true" (or any value) to enable, empty to disable
USE_LORA=true


# Tweet Features
MAX_TWEETS=50
TWEET_TEXT_MODEL=""        # Leave empty to reuse TEXT_MODEL
TWEET_TEXT_MAX_LEN=128
TWEET_TEXT_BATCH=512
ENABLE_TWEET_TEXT=true

# Evaluation
EVAL_SPLIT=test

# Preprocessing
REBUILD_FLAG="--rebuild"            # Set to "--rebuild" to force preprocessing rebuild
# REBUILD_FLAG="--rebuild"

RESUME_FLAG=""             
# RESUME_FLAG="--resume"


echo "=============================================="
echo "TwiBot-22 Training Configuration"
echo "=============================================="
echo "Model: HIDDEN_DIM=$HIDDEN_DIM, LAYERS=$NUM_LAYERS, HEADS=$GRAPH_HEADS"
echo "Training: BATCH=$BATCH_SIZE, LR_BACKBONE=$LR_BACKBONE, LR_TEXT=$LR_TEXT"
echo "Regularization: DROPOUT=$DROPOUT, WEIGHT_DECAY=$WEIGHT_DECAY, LABEL_SMOOTHING=$LABEL_SMOOTHING"
echo "Class Balance: BOT_WEIGHT_MULT=$BOT_WEIGHT_MULT"
echo "Graph: NUM_NEIGHBORS=$NUM_NEIGHBORS"
echo "Text: MODEL=$TEXT_MODEL, TRAINABLE=$TEXT_TRAINABLE, LORA=$USE_LORA"
echo "=============================================="

TEXT_TRAIN_FLAG="${TEXT_TRAINABLE:+--text-trainable}"
LORA_FLAG="${USE_LORA:+--use-lora}"
TWEET_TEXT_ARGS="${ENABLE_TWEET_TEXT:+--enable-tweet-text --tweet-text-model ${TWEET_TEXT_MODEL:-$TEXT_MODEL} --tweet-text-max-length $TWEET_TEXT_MAX_LEN --tweet-text-batch-size $TWEET_TEXT_BATCH}"
TARGET_POS_ARGS="${TARGET_POS_RATIO:+--target-pos-ratio $TARGET_POS_RATIO}"

echo ""
echo "==> Starting training..."
python train.py \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --graph-heads "$GRAPH_HEADS" \
    --dropout "$DROPOUT" \
    --num-neighbors $NUM_NEIGHBORS \
    --max-tweets "$MAX_TWEETS" \
    --text-max-length "$TEXT_MAX_LEN" \
    --text-model "$TEXT_MODEL" \
    --loss focal \
    --gamma 1.5 \
    --bot-weight-mult "$BOT_WEIGHT_MULT" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --lr-backbone "$LR_BACKBONE" \
    --lr-text "$LR_TEXT" \
    --weight-decay "$WEIGHT_DECAY" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --use-hgt \
    $TEXT_TRAIN_FLAG \
    $LORA_FLAG \
    $TWEET_TEXT_ARGS \
    $TARGET_POS_ARGS \
    $REBUILD_FLAG \
    $RESUME_FLAG

echo ""
echo "==> Evaluating on $EVAL_SPLIT split..."
python eval.py \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --graph-heads "$GRAPH_HEADS" \
    --dropout "$DROPOUT" \
    --num-neighbors $NUM_NEIGHBORS \
    --max-tweets "$MAX_TWEETS" \
    --text-max-length "$TEXT_MAX_LEN" \
    --text-model "$TEXT_MODEL" \
    --loss focal \
    --gamma 1.5 \
    --bot-weight-mult "$BOT_WEIGHT_MULT" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --split "$EVAL_SPLIT" \
    --use-hgt \
    $TEXT_TRAIN_FLAG \
    $LORA_FLAG \
    $TWEET_TEXT_ARGS

echo ""
echo "==> Done!"
