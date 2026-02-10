#!/usr/bin/env bash
set -euo pipefail

log() { echo "[preflight] $*"; }
die() { echo "[preflight] ERROR: $*" >&2; exit 2; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  die "Not running inside a Slurm job (SLURM_JOB_ID is empty)."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. Are you on a GPU node with NVIDIA drivers?"
fi

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${gpu_count}" -le 0 ]]; then
  die "No GPUs detected on this node."
fi

min_vram_gb="${MIN_VRAM_GB_PER_GPU:-0}"
if ! [[ "${min_vram_gb}" =~ ^[0-9]+$ ]]; then
  die "MIN_VRAM_GB_PER_GPU must be an integer (got: ${min_vram_gb})."
fi

log "node=$(hostname)"
log "gpu_count=${gpu_count}"

# Query memory.total in MiB, then compute the minimum across GPUs.
min_mem_mib="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | awk 'BEGIN{min=-1} {if(min<0 || $1<min) min=$1} END{print min}'
)"

if [[ -z "${min_mem_mib}" ]]; then
  die "Could not query GPU memory via nvidia-smi."
fi

min_mem_gb="$(( (min_mem_mib + 1023) / 1024 ))"
log "min_vram_gb_detected=${min_mem_gb} (min across GPUs)"

if [[ "${min_vram_gb}" -gt 0 && "${min_mem_gb}" -lt "${min_vram_gb}" ]]; then
  die "Insufficient VRAM: detected ${min_mem_gb} GB < required ${min_vram_gb} GB. Refusing to start server."
fi

if [[ -n "${ENV_PREFIX:-}" && -x "${ENV_PREFIX}/bin/python" ]]; then
  PY="${ENV_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  log "python not found in PATH (server launch will likely fail)."
  PY=""
fi

if [[ -n "${PY}" ]]; then
  "${PY}" -V 2>&1 | sed 's/^/[preflight] /'
  if [[ "${SKIP_VLLM_CHECK:-0}" == "1" ]]; then
    log "SKIP_VLLM_CHECK=1; skipping vllm import check"
  else
    if "${PY}" -c 'import vllm' >/dev/null 2>&1; then
      log "vllm import OK"
    else
      die "vllm is not importable in this environment. Load/activate your vLLM env in the sbatch script (or set SKIP_VLLM_CHECK=1)."
    fi
  fi
fi

log "preflight OK"
