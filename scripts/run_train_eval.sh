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

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Setup
cd /sciclone/home/hwhyman/Graph_learning
source .venv/bin/activate
mkdir -p logs

# ============ CONFIGURATION ============

HIDDEN_DIM=256
NUM_LAYERS=3
GRAPH_HEADS=4
DROPOUT=0.3            

BATCH_SIZE=128             
EPOCHS=15
PATIENCE=5
LR_BACKBONE=1e-4         
LR_TEXT=2e-5              
WEIGHT_DECAY=0.05        
MAX_GRAD_NORM=1.0

BOT_WEIGHT_MULT=2.0       
TARGET_POS_RATIO=""        
LABEL_SMOOTHING=0.1        

NUM_NEIGHBORS="20 10 5"    

TEXT_MODEL="xlm-roberta-base"
TEXT_MAX_LEN=256         
TEXT_TRAINABLE=""
USE_LORA=true

MAX_TWEETS=50
TWEET_TEXT_MODEL=""
TWEET_TEXT_MAX_LEN=128
TWEET_TEXT_BATCH=512
ENABLE_TWEET_TEXT=true

EVAL_SPLIT=test

#REBUILD_FLAG="--rebuild"
REBUILD_FLAG=""

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
