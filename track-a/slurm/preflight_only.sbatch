#!/bin/bash -l
#SBATCH -J glm47-preflight
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:10
#SBATCH -o preflight-%j.out
#SBATCH -e preflight-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

echo "[preflight-only] job_id=${SLURM_JOB_ID:-}"
echo "[preflight-only] nodelist=${SLURM_JOB_NODELIST:-}"
echo "[preflight-only] starting at $(date)"
echo "[preflight-only] NOTE: override partition with: sbatch -p <partition> this_script.sbatch"
echo "[preflight-only] repo_dir=${REPO_DIR}"

# ---- Environment (edit as needed) ----
# If your cluster needs modules/conda to see CUDA/nvidia-smi/python, set them here:
# module load cuda-11.0.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

# Preflight thresholds (tune once you know your GPU type)
export MIN_VRAM_GB_PER_GPU="${MIN_VRAM_GB_PER_GPU:-48}"

# Optional: bootstrap conda env + install vLLM (slow; use only if you want to test vLLM import)
export BOOTSTRAP_CONDA_ENV="${BOOTSTRAP_CONDA_ENV:-0}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export ENV_PREFIX="${ENV_PREFIX:-${SCRATCH:-$HOME}/.conda_envs/glm47-vllm-py310}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_DIR}/track-a/requirements.txt}"

# If you only want GPU/VRAM validation and don't yet have vLLM installed, set:
export SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"

if [[ ! -x "${REPO_DIR}/track-a/bin/preflight.sh" ]]; then
  echo "[preflight-only] ERROR: ${REPO_DIR}/track-a/bin/preflight.sh not found/executable." >&2
  echo "[preflight-only] Submit from the repo root so SLURM_SUBMIT_DIR points to it, e.g.:" >&2
  echo "[preflight-only]   cd /path/to/dop-project2 && sbatch track-a/slurm/preflight_only.sbatch" >&2
  echo "[preflight-only] Or set REPO_DIR explicitly:" >&2
  echo "[preflight-only]   REPO_DIR=/path/to/dop-project2 sbatch track-a/slurm/preflight_only.sbatch" >&2
  exit 2
fi

cd "${REPO_DIR}"
if [[ "${BOOTSTRAP_CONDA_ENV}" == "1" ]]; then
  echo "[preflight-only] bootstrapping conda env: ${ENV_PREFIX} (python=${PYTHON_VERSION})"
  "${REPO_DIR}/track-a/bin/bootstrap_conda_env.sh"
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  echo "[preflight-only] using python: $(python -V 2>&1)"
fi

srun --export=ALL --ntasks=1 "${REPO_DIR}/track-a/bin/preflight.sh"
