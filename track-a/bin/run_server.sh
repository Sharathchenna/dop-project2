#!/usr/bin/env bash
set -euo pipefail

log() { echo "[track-a] $*"; }
die() { echo "[track-a] ERROR: $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-4.7-Flash}"
MODEL_ALIAS="${MODEL_ALIAS:-glm47-flash30b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

log "node=$(hostname)"
log "job_id=${SLURM_JOB_ID:-}"
log "bind=${HOST}:${PORT}"

"${ROOT_DIR}/track-a/bin/preflight.sh"

if [[ -n "${ENV_PREFIX:-}" && -x "${ENV_PREFIX}/bin/python" ]]; then
  PY="${ENV_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  die "python/python3 not found in PATH."
fi

# Prefer explicit override, because vLLM entrypoints can differ across versions.
if [[ -n "${VLLM_SERVER_CMD:-}" ]]; then
  log "starting server via VLLM_SERVER_CMD"
  log "cmd=${VLLM_SERVER_CMD}"
  exec bash -lc "${VLLM_SERVER_CMD}"
fi

log "starting vLLM OpenAI-compatible server"

# NOTE: This entrypoint is common for vLLM, but if your version differs, set VLLM_SERVER_CMD.
exec "${PY}" -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_ID}" \
  --served-model-name "${MODEL_ALIAS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
