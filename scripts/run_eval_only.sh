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

# Activate the virtual environment
source /sciclone/home/hwhyman/Graph_learning/.venv/bin/activate

PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_ACTIVATE}"
fi

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
fi

cd /sciclone/home/hwhyman/Graph_learning

mkdir -p logs

RAW_DATA=${RAW_DATA:-TwiBot-22}
PROCESSED_DIR=${PROCESSED_DIR:-processed}
DEVICE=${DEVICE:-cuda}
EVAL_SPLIT=${EVAL_SPLIT:-test}

missing_modules=()
for module in torch transformers torch_geometric; do
    if ! "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
        missing_modules+=("${module}")
    fi
done

if (( ${#missing_modules[@]} > 0 )); then
    echo "[ERROR] Missing Python packages: ${missing_modules[*]}" >&2
    echo "Install dependencies (e.g. pip install -r requirements.txt) before running." >&2
    exit 1
fi

read -r -a EVAL_ARGS <<< "${EVAL_ARGS:-}"

"${PYTHON_BIN}" eval.py --raw-data "${RAW_DATA}" --processed-dir "${PROCESSED_DIR}" --device "${DEVICE}" --split "${EVAL_SPLIT}" "${EVAL_ARGS[@]}"

