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
# Default to $HOME to avoid clusters where $SCRATCH is not exported in all contexts.
# Override to a scratch filesystem explicitly if desired.
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda_envs/glm47-vllm-py310}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-}"
# If you want the script to run `conda tos accept ...` for the default Anaconda channels,
# you must opt in explicitly.
AUTO_ACCEPT_ANACONDA_TOS="${AUTO_ACCEPT_ANACONDA_TOS:-0}"

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

maybe_accept_tos() {
  if [[ "${AUTO_ACCEPT_ANACONDA_TOS}" != "1" ]]; then
    return 0
  fi

  # This subcommand exists only in newer conda. Best-effort.
  if conda tos --help >/dev/null 2>&1; then
    log "AUTO_ACCEPT_ANACONDA_TOS=1; accepting Anaconda channel ToS (main, r)"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  fi
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

  maybe_accept_tos

  log "creating conda env at ${ENV_PREFIX} with python=${PYTHON_VERSION}"
  if command -v mamba >/dev/null 2>&1; then
    mamba create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
  else
    conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
  fi
}

pip_install() {
  local pybin="${ENV_PREFIX}/bin/python"
  if [[ ! -x "${pybin}" ]]; then
    die "expected python at ${pybin} but it does not exist/executable"
  fi

  log "python=$("${pybin}" -V 2>&1)"
  log "pip=$("${pybin}" -m pip -V 2>&1)"

  "${pybin}" -m pip install -U pip setuptools wheel

  # vLLM frequently falls back to building from source when no matching wheel exists.
  # Source builds require a CUDA toolkit (nvcc) on PATH.
  if [[ -n "${REQUIREMENTS_FILE}" && -f "${REQUIREMENTS_FILE}" ]]; then
    if rg -q -n '^vllm([<=>].*)?$' "${REQUIREMENTS_FILE}" 2>/dev/null; then
      if ! command -v nvcc >/dev/null 2>&1; then
        log "WARNING: nvcc not found on PATH. If pip cannot find a prebuilt vLLM wheel, it will try to compile and fail."
        log "WARNING: load a CUDA toolkit module (with nvcc) before running this script, or use a wheelhouse."
      fi
    fi
  fi

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
    "${pybin}" -m pip install "${pip_args[@]}" -r "${REQUIREMENTS_FILE}"
  else
    log "installing packages: ${BOOTSTRAP_PACKAGES[*]}"
    "${pybin}" -m pip install "${pip_args[@]}" "${BOOTSTRAP_PACKAGES[@]}"
  fi

  "${pybin}" -c 'import vllm; print("vllm_ok", vllm.__version__)' >/dev/null 2>&1 \
    && log "vllm import OK" \
    || die "vllm import failed after install. This is usually a CUDA/Torch mismatch; load the correct CUDA module and/or pin torch/vllm versions."
}

main() {
  ensure_conda
  create_env_if_needed
  pip_install
}

main "$@"
