#!/bin/bash
#SBATCH --job-name=twibot_eda
#SBATCH --output=logs/twibot_train_eval_%j.log
#SBATCH --error=logs/twibot_train_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# SBATCH --gres=gpu:1  # Uncomment if you want to run embeddings on GPU

set -euo pipefail

WORKDIR="/sciclone/home/hwhyman/Graph_learning"
VENV_DIR="$WORKDIR/.venv"

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# If your cluster uses environment modules, load Python here. Example:
# module purge
# module load python/3.9

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade -r requirements.txt

python run_eda.py
