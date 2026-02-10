#!/bin/bash -l
#SBATCH -J glm47-track-a
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-23:50
#SBATCH -o track-a-%j.out
#SBATCH -e track-a-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[track-a] job_id=${SLURM_JOB_ID:-}"
echo "[track-a] user=${USER:-}"
echo "[track-a] submit_host=${SLURM_SUBMIT_HOST:-}"
echo "[track-a] nodelist=${SLURM_JOB_NODELIST:-}"
echo "[track-a] starting at $(date)"
echo "[track-a] NOTE: override partition with: sbatch -p <partition> this_script.sbatch"
echo "[track-a] script_dir=${SCRIPT_DIR}"
echo "[track-a] repo_dir=${REPO_DIR}"

# ---- Environment (edit as needed) ----
# You must ensure vLLM is available on the compute node:
# Examples (uncomment/adapt):
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

export MODEL_ID="${MODEL_ID:-GLM-4.7-Flash-30B}"          # HF model id or local path
export MODEL_ALIAS="${MODEL_ALIAS:-glm47-flash30b}"       # stable client-facing name

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Preflight gate. Default is conservative and may need adjustment once you know your GPUs.
export MIN_VRAM_GB_PER_GPU="${MIN_VRAM_GB_PER_GPU:-48}"

# vLLM knobs (optional)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Cache locations (set to a persistent FS if available; avoid relying on $SCRATCH unless intentional)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

echo "[track-a] model_id=${MODEL_ID}"
echo "[track-a] model_alias=${MODEL_ALIAS}"
echo "[track-a] bind=${HOST}:${PORT}"
echo "[track-a] min_vram_gb_per_gpu=${MIN_VRAM_GB_PER_GPU}"
echo "[track-a] tp_size=${TENSOR_PARALLEL_SIZE}"

echo "[track-a] launching server under srun..."

# Use srun to comply with cluster policy (run inside allocation).
cd "${REPO_DIR}"
srun --ntasks=1 "${REPO_DIR}/track-a/bin/run_server.sh"
