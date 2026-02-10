#!/bin/bash -l
#SBATCH -J glm47-preflight
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:10
#SBATCH -o preflight-%j.out
#SBATCH -e preflight-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[preflight-only] job_id=${SLURM_JOB_ID:-}"
echo "[preflight-only] nodelist=${SLURM_JOB_NODELIST:-}"
echo "[preflight-only] starting at $(date)"
echo "[preflight-only] NOTE: override partition with: sbatch -p <partition> this_script.sbatch"
echo "[preflight-only] script_dir=${SCRIPT_DIR}"
echo "[preflight-only] repo_dir=${REPO_DIR}"

# ---- Environment (edit as needed) ----
# If your cluster needs modules/conda to see CUDA/nvidia-smi/python, set them here:
# module load cuda-11.0.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

# Preflight thresholds (tune once you know your GPU type)
export MIN_VRAM_GB_PER_GPU="${MIN_VRAM_GB_PER_GPU:-48}"

# If you only want GPU/VRAM validation and don't yet have vLLM installed, set:
export SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"

cd "${REPO_DIR}"
srun --ntasks=1 "${REPO_DIR}/track-a/bin/preflight.sh"
