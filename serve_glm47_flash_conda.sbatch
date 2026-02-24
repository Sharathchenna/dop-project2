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
# Verified against official Unsloth guide:
#   https://unsloth.ai/docs/models/glm-4.7-flash#glm-4.7-flash-in-vllm
#
# Key facts from Unsloth guide:
#   - vLLM nightly MUST use the cu130 index: https://wheels.vllm.ai/nightly/cu130
#     (this is what makes glm47 tool-call-parser available)
#   - Use Unsloth's FP8 model: unsloth/GLM-4.7-Flash-FP8-Dynamic
#     (50% less KV cache memory vs base model, fits single H100 comfortably)
#   - numba must be installed explicitly
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False is required
#   - --tool-call-parser glm47 and --reasoning-parser glm45 are both correct
#   - For 1 GPU: CUDA_VISIBLE_DEVICES='0', --tensor-parallel-size 1
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

# 3. Upgrade pip tooling
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch for CUDA 12.1
#    Check your cluster CUDA version with: nvcc --version
#    If CUDA 12.3+, switch to: https://download.pytorch.org/whl/cu124
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Install latest vLLM nightly using the cu130 index.
#    CRITICAL: cu130 is the correct index — using the bare /nightly URL gives
#    an older build that is missing the glm47 tool-call parser.
pip install \
    --no-cache-dir \
    --force-reinstall \
    --upgrade \
    --pre \
    vllm \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130

# 6. Install transformers from GitHub tarball (no git required on cluster).
#    Must come AFTER vLLM to override its pinned transformers version.
pip install --no-cache-dir --force-reinstall --upgrade \
    "https://github.com/huggingface/transformers/archive/refs/heads/main.tar.gz"

# 7. Install numba (required by vLLM nightly per Unsloth guide)
pip install --no-cache-dir --force-reinstall numba

# 8. Fix huggingface_hub to be compatible with transformers 5.x
pip install --no-cache-dir "huggingface_hub>=1.0,<2.0"

# 9. Diagnostics
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub", "numba"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: ERROR:", e)

# Verify glm47 parser is present — if missing, nightly wheel is still too old
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    parsers = sorted(ToolParserManager._tool_parsers.keys())
    print("Tool parsers available:", parsers)
    if "glm47" in parsers:
        print("glm47 parser: OK")
    else:
        print("FATAL: glm47 parser still missing — check nightly index URL")
        sys.exit(1)
except Exception as e:
    print("Parser check error:", e)
    sys.exit(1)
PY
nvidia-smi

# 10. Serve GLM-4.7-Flash using Unsloth's FP8-Dynamic model
#
#    Why unsloth/GLM-4.7-Flash-FP8-Dynamic instead of zai-org/GLM-4.7-Flash:
#      - FP8 quantization reduces KV cache memory usage by ~50%
#      - Allows larger max-model-len on a single H100
#      - Officially recommended by Unsloth for vLLM deployments
#
#    Flag reference (all verified against Unsloth guide):
#      --tool-call-parser glm47           GLM-4.7 specific tool call parser
#      --reasoning-parser glm45           GLM-4.x thinking/reasoning parser
#      --enable-auto-tool-choice          exposes tool-calling via OpenAI API
#      --quantization fp8                 enable FP8 weight quantization
#      --kv-cache-dtype fp8               enable FP8 KV cache (50% memory saving)
#      --max_num_batched_tokens 16384     from Unsloth guide
#      CUDA_VISIBLE_DEVICES='0'           single GPU (Unsloth uses 4; we use 1)
#      PYTORCH_CUDA_ALLOC_CONF=...        required to avoid CUDA allocator issues

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

CUDA_VISIBLE_DEVICES='0' stdbuf -oL -eL vllm serve unsloth/GLM-4.7-Flash-FP8-Dynamic \
  --served-model-name glm-4.7-flash \
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
