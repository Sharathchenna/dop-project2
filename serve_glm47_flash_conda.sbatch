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
# Sources verified against:
#   - Official vLLM tool calling docs: https://docs.vllm.ai/en/latest/features/tool_calling/
#   - Official vLLM recipes GLM guide: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html
#   - Official vLLM recipes GLM-5 guide (uses same install method + glm47 parser)
#
# ROOT CAUSE OF ALL PREVIOUS FAILURES:
#   glm47 parser just merged into vLLM and missed the 0.13.0 stable release.
#   /nightly and /nightly/cu130 index URLs serve cached old builds without glm47.
#   Fix: install from a SPECIFIC COMMIT HASH that post-dates the glm47 merge.
#   The commit below is taken from the official vLLM GLM-5 recipe and is known
#   to include the glm47 parser.
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

# 3. Install uv if not already present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 4. Install vLLM from a SPECIFIC COMMIT that includes glm47 parser.
#    This is the official method from the vLLM GLM-5 recipe guide.
#    Using commit hash instead of /nightly guarantees the correct build.
export VLLM_COMMIT=ec12d39d44739bee408ec1473acc09e75daf1a5d

uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}

# 5. Install transformers from GitHub source (uv handles git natively,
#    no system git required)
uv pip install "git+https://github.com/huggingface/transformers.git"

# 6. Install numba (required per official Unsloth + vLLM GLM guides)
uv pip install --force-reinstall numba

# 7. Diagnostics — job exits here if glm47 is still missing
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub", "numba"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: ERROR:", e)

try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    parsers = sorted(ToolParserManager._tool_parsers.keys())
    print("Tool parsers:", parsers)
    assert "glm47" in parsers, "FATAL: glm47 parser missing — wrong commit build"
    print("glm47: OK")
except AssertionError as e:
    print(e); sys.exit(1)
except Exception as e:
    print("Parser check error:", e); sys.exit(1)
PY
nvidia-smi

# 8. Serve GLM-4.7-Flash
#
#    Parser reference (from official vLLM tool calling docs):
#      GLM-4.5, GLM-4.5-Air, GLM-4.6  →  --tool-call-parser glm45
#      GLM-4.7, GLM-4.7-Flash          →  --tool-call-parser glm47  ← correct
#      All GLM-4.x reasoning            →  --reasoning-parser glm45
#
#    Using Unsloth FP8-Dynamic model: 50% less KV cache memory vs base model
#    Adapted for 1x H100 (official guide targets 4x GPU):
#      --tensor-parallel-size 1 and --max-model-len 8192

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
  --no-enable-prefix-caching \
  --disable-log-requests \
  --host 0.0.0.0 \
  --port 8000
