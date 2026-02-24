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
# Notes:
# - GLM-4.7-Flash requires vLLM nightly build + transformers from GitHub source.
# - Do NOT use stable PyPI vLLM or --no-deps; the model uses glm4_moe_lite arch
#   which is only in transformers@main.
# - --tokenizer-mode slow is required to avoid AttributeError on all_special_tokens_extended.
# - --tool-call-parser glm47 + --reasoning-parser glm45 are required for correct
#   tool-use and thinking mode behaviour per official zai-org docs.
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

# 4. Install PyTorch (CUDA 12.1 wheel — adjust index if your cluster uses 12.4+)
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Install vLLM nightly (required for glm4_moe_lite architecture support)
#    Do NOT use --no-deps; let vLLM resolve its own dependency pins.
pip install -U vllm \
    --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# 6. Install transformers from GitHub main AFTER vLLM
#    (provides glm4_moe_lite model config not yet on PyPI release)
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install --upgrade huggingface_hub

# 7. Diagnostics
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED or import error:", e)
PY
nvidia-smi

# 8. Serve GLM-4.7-Flash
#    Key flags explained:
#      --tokenizer-mode slow        : fixes AttributeError on all_special_tokens_extended
#      --tool-call-parser glm47     : correct tool-call parsing for this model family
#      --reasoning-parser glm45     : correct thinking/reasoning mode parsing
#      --enable-auto-tool-choice    : exposes tool-calling via OpenAI-compatible API
#      --no-enable-prefix-caching   : recommended by official vLLM GLM recipe
#      --disable-log-requests       : reduces log noise in production
#      SAFETENSORS_FAST_GPU=1       : recommended by official vLLM GLM recipe

SAFETENSORS_FAST_GPU=1 stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-mode slow \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --host 0.0.0.0 \
  --port 8000
