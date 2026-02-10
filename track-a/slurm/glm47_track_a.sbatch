#!/bin/bash -l
#SBATCH -J glm47-track-a
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-23:50
#SBATCH -o track-a-%j.out
#SBATCH -e track-a-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

echo "[track-a] job_id=${SLURM_JOB_ID:-}"
echo "[track-a] user=${USER:-}"
echo "[track-a] submit_host=${SLURM_SUBMIT_HOST:-}"
echo "[track-a] nodelist=${SLURM_JOB_NODELIST:-}"
echo "[track-a] starting at $(date)"
echo "[track-a] NOTE: override partition with: sbatch -p <partition> this_script.sbatch"
echo "[track-a] repo_dir=${REPO_DIR}"

# ---- Environment (edit as needed) ----
# You must ensure vLLM is available on the compute node:
# Examples (uncomment/adapt):
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

export MODEL_ID="${MODEL_ID:-zai-org/GLM-4.7-Flash}"      # HF model id or local path
export MODEL_ALIAS="${MODEL_ALIAS:-glm47-flash30b}"       # stable client-facing name

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Preflight gate. Default is conservative and may need adjustment once you know your GPUs.
export MIN_VRAM_GB_PER_GPU="${MIN_VRAM_GB_PER_GPU:-48}"

# vLLM knobs (optional)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# ---- Conda bootstrap (optional) ----
# The cluster banner requests package installation (anaconda3) be done from an interactive Slurm shell.
# So default is disabled; create the env interactively, then run the server job using that ENV_PREFIX.
# You can still set BOOTSTRAP_CONDA_ENV=1 if your policy/admin allows installs in batch jobs.
export BOOTSTRAP_CONDA_ENV="${BOOTSTRAP_CONDA_ENV:-0}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export ENV_PREFIX="${ENV_PREFIX:-${SCRATCH:-$HOME}/.conda_envs/glm47-vllm-py310}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_DIR}/track-a/requirements.txt}"

# Cache locations (set to a persistent FS if available; avoid relying on $SCRATCH unless intentional)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

echo "[track-a] model_id=${MODEL_ID}"
echo "[track-a] model_alias=${MODEL_ALIAS}"
echo "[track-a] bind=${HOST}:${PORT}"
echo "[track-a] min_vram_gb_per_gpu=${MIN_VRAM_GB_PER_GPU}"
echo "[track-a] tp_size=${TENSOR_PARALLEL_SIZE}"

if [[ "${BOOTSTRAP_CONDA_ENV}" == "1" ]]; then
  echo "[track-a] bootstrapping conda env: ${ENV_PREFIX} (python=${PYTHON_VERSION})"
  "${REPO_DIR}/track-a/bin/bootstrap_conda_env.sh"
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  echo "[track-a] using python: $(python -V 2>&1)"
else
  echo "[track-a] WARNING: ${ENV_PREFIX}/bin/python not found; will use system python on PATH." >&2
fi

echo "[track-a] launching server under srun..."

# Use srun to comply with cluster policy (run inside allocation).
if [[ ! -x "${REPO_DIR}/track-a/bin/run_server.sh" ]]; then
  echo "[track-a] ERROR: ${REPO_DIR}/track-a/bin/run_server.sh not found/executable." >&2
  echo "[track-a] Submit from the repo root so SLURM_SUBMIT_DIR points to it, or set REPO_DIR explicitly." >&2
  exit 2
fi

# If you submit this sbatch from inside another Slurm allocation (interactive `srun` shell),
# Slurm environment variables can leak and break nested srun steps.
unset SLURM_TRES_PER_TASK || true

# Some clusters set a default CPU binding policy that can fail for nested steps.
# Force no CPU binding for the step to avoid:
# "Unable to satisfy cpu bind request"
export SLURM_CPU_BIND=none
unset SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE || true

cd "${REPO_DIR}"
srun --cpu-bind=none --ntasks=1 "${REPO_DIR}/track-a/bin/run_server.sh"
