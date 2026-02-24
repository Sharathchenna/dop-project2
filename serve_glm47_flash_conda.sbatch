#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --job-name=glm47_flash_1gpu
#SBATCH -o glm47_flash_1gpu.%j.out
#SBATCH -e glm47_flash_1gpu.%j.err
# ---------------------------
# Install method matches official Unsloth guide exactly:
#   https://unsloth.ai/docs/models/glm-4.7-flash#glm-4.7-flash-in-vllm
#
# Key: uses "uv pip install" not "pip install"
#   - uv has built-in git support, so git+https:// works without system git
#   - uv correctly resolves the nightly wheel that includes glm47 parser
#   - uv is faster and more reliable for nightly/pre-release installs
#
# Adapted for 1x H100 (Unsloth guide targets 4x GPU):
#   - CUDA_VISIBLE_DEVICES='0' instead of '0,1,2,3'
#   - --tensor-parallel-size 1 instead of 4
#   - --max-model-len 8192 instead of 200000 (200K ctx needs 4 GPUs)
# ---------------------------
set -x

# 1. Activate conda env
source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm
export PYTHONUNBUFFERED=1

# 2. Hugging Face cache
export HF_HOME=$HOME/.cache/huggingface
mkdir -p "$HF_HOME"

# 3. Install uv if not already present (no sudo needed, installs to ~/.local)
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv found: $(uv --version)"
    export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Install PyTorch for CUDA 12.1
#    uv pip install respects the active conda env
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Install vLLM nightly using uv exactly as Unsloth guide specifies
#    --torch-backend=auto lets uv detect the correct CUDA torch variant
#    --extra-index-url https://wheels.vllm.ai/nightly/cu130 is the specific
#    nightly index that contains the glm47 tool-call parser
uv pip install \
    --upgrade \
    --force-reinstall \
    vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130

# 6. Install transformers from GitHub using uv (works without system git)
uv pip install \
    --upgrade \
    --force-reinstall \
    "git+https://github.com/huggingface/transformers.git"

# 7. Install numba (required per Unsloth guide)
uv pip install --force-reinstall numba

# 8. Diagnostics
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub", "numba"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: ERROR:", e)

# Confirm glm47 parser is present — job will exit here if still missing
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    parsers = sorted(ToolParserManager._tool_parsers.keys())
    print("Tool parsers:", parsers)
    assert "glm47" in parsers, "FATAL: glm47 parser missing — wrong nightly build"
    print("glm47 parser: OK")
except AssertionError as e:
    print(e); sys.exit(1)
except Exception as e:
    print("Parser check error:", e); sys.exit(1)
PY
nvidia-smi

# 9. Serve GLM-4.7-Flash (1 GPU version of official Unsloth command)
#
#    Official Unsloth 4-GPU command:
#      CUDA_VISIBLE_DEVICES='0,1,2,3' ... --tensor-parallel-size 4 --max-model-len 200000
#
#    Adapted for 1x H100:
#      CUDA_VISIBLE_DEVICES='0'       single GPU
#      --tensor-parallel-size 1       match GPU count
#      --max-model-len 8192           reduced from 200K (needs 4 GPUs for full ctx)
#
#    Note: PYTORCH_ALLOC_CONF (not PYTORCH_CUDA_ALLOC_CONF — that name is deprecated)

export PYTORCH_ALLOC_CONF=expandable_segments:False

CUDA_VISIBLE_DEVICES='0' stdbuf -oL -eL vllm serve unsloth/GLM-4.7-Flash-FP8-Dynamic \
  --served-model-name unsloth/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --dtype bfloat16 \
  --seed 3407 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --max_num_batched_tokens 16384 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --disable-log-requests \
  --host 0.0.0.0 \
  --port 8000
