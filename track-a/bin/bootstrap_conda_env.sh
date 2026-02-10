#!/usr/bin/env bash
set -euo pipefail

log() { echo "[bootstrap] $*"; }
die() { echo "[bootstrap] ERROR: $*" >&2; exit 2; }

# This script is meant to run inside a Slurm job before launching the server.
# It creates (once) and reuses (subsequent runs) a conda env at ENV_PREFIX.
#
# Notes:
# - On many HPC systems, outbound internet is restricted. If pip cannot download wheels,
#   set WHEELHOUSE_DIR to a directory containing pre-downloaded wheels and requirements.
# - vLLM has tight coupling to CUDA/Torch; you may need to load the right CUDA module first.

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ENV_PREFIX="${ENV_PREFIX:-${SCRATCH:-$HOME}/.conda_envs/glm47-vllm-py310}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-}"

BOOTSTRAP_PACKAGES_DEFAULT=("vllm")
IFS=',' read -r -a BOOTSTRAP_PACKAGES <<< "${BOOTSTRAP_PACKAGES:-}"
if [[ "${#BOOTSTRAP_PACKAGES[@]}" -eq 0 || -z "${BOOTSTRAP_PACKAGES[0]}" ]]; then
  BOOTSTRAP_PACKAGES=("${BOOTSTRAP_PACKAGES_DEFAULT[@]}")
fi

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  # Allow user to point at conda.sh explicitly.
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    command -v conda >/dev/null 2>&1 && return 0
  fi

  # Common user installs.
  for csh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    ; do
    if [[ -f "${csh}" ]]; then
      # shellcheck disable=SC1090
      source "${csh}"
      command -v conda >/dev/null 2>&1 && return 0
    fi
  done

  die "conda not found. Load your anaconda/miniconda module in the sbatch script or install Miniconda under \$HOME and set CONDA_SH=/path/to/conda.sh."
}

create_env_if_needed() {
  if [[ -d "${ENV_PREFIX}" && -f "${ENV_PREFIX}/bin/python" ]]; then
    log "env exists: ${ENV_PREFIX}"
    return 0
  fi

  mkdir -p "$(dirname "${ENV_PREFIX}")"

  # Simple lock to avoid two jobs racing to create the same env.
  local lockdir="${ENV_PREFIX}.lockdir"
  local waited=0
  while ! mkdir "${lockdir}" 2>/dev/null; do
    waited=$((waited + 1))
    if [[ "${waited}" -gt 120 ]]; then
      die "Timed out waiting for env lock (${lockdir})."
    fi
    sleep 2
  done
  trap 'rmdir "${lockdir}" >/dev/null 2>&1 || true' EXIT

  if [[ -d "${ENV_PREFIX}" && -f "${ENV_PREFIX}/bin/python" ]]; then
    log "env created by another job while waiting: ${ENV_PREFIX}"
    return 0
  fi

  log "creating conda env at ${ENV_PREFIX} with python=${PYTHON_VERSION}"
  if command -v mamba >/dev/null 2>&1; then
    mamba create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
  else
    conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
  fi
}

activate_env() {
  # conda activate requires conda.sh to be sourced. ensure_conda already did that if needed.
  # shellcheck disable=SC1091
  if [[ -z "${CONDA_SH:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
  fi
  # Best-effort: conda should now be a function.
  conda activate "${ENV_PREFIX}" >/dev/null 2>&1 || conda activate "${ENV_PREFIX}"
}

pip_install() {
  log "python=$(python -V 2>&1)"
  log "pip=$(python -m pip -V 2>&1)"

  python -m pip install -U pip setuptools wheel

  local pip_args=()
  if [[ -n "${WHEELHOUSE_DIR}" ]]; then
    pip_args+=(--no-index --find-links "${WHEELHOUSE_DIR}")
    log "using wheelhouse: ${WHEELHOUSE_DIR}"
  fi

  if [[ -n "${REQUIREMENTS_FILE}" ]]; then
    if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
      die "REQUIREMENTS_FILE not found: ${REQUIREMENTS_FILE}"
    fi
    log "installing requirements: ${REQUIREMENTS_FILE}"
    python -m pip install "${pip_args[@]}" -r "${REQUIREMENTS_FILE}"
  else
    log "installing packages: ${BOOTSTRAP_PACKAGES[*]}"
    python -m pip install "${pip_args[@]}" "${BOOTSTRAP_PACKAGES[@]}"
  fi

  python -c 'import vllm; print("vllm_ok", vllm.__version__)' >/dev/null 2>&1 \
    && log "vllm import OK" \
    || die "vllm import failed after install. This is usually a CUDA/Torch mismatch; load the correct CUDA module and/or pin torch/vllm versions."
}

main() {
  ensure_conda
  create_env_if_needed
  activate_env
  pip_install
}

main "$@"

