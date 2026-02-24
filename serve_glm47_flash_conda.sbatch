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
# - This script intentionally uses prebuilt vllm wheel (no compilation).
# - Transformers from GitHub must be installed LAST to ensure glm4_moe_lite is available.
# - Codecademy suggests using --trust-remote-code and --dtype bfloat16 for vllm serve. (See source)
# ---------------------------
set -x
# do not use "set -e" blindly; we want visible failure in output but not silent exits
# ---------------------------
# 0. Optional: set HF token (recommended to avoid HF rate limits)
# export HF_TOKEN="hf_...."

# 1. Activate conda env
source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONUNBUFFERED=1

# 2. Hugging Face cache (safe user location)
export HF_HOME=$HOME/.cache/huggingface
mkdir -p "$HF_HOME"

# 3. Clean previous partial installs (safe)
pip uninstall -y vllm transformers huggingface-hub || true

# 4. Upgrade build tooling
pip install --upgrade pip setuptools wheel

# 5. Install CUDA-capable PyTorch (use the cu121 index for CUDA 12.1)
#    Adjust if your cluster needs a different CUDA wheel
pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121

# 6. Install vLLM from PyPI (prebuilt wheel). Use --no-deps to avoid it downgrading transformers.
#    If a more recent prebuilt wheel exists (e.g. --pre) you can use that variant.
pip install --upgrade vllm --no-deps

# 7. Install Transformers & huggingface_hub from GitHub main (MUST BE LAST)
#    This provides the new glm4_moe_lite model config.
pip install --upgrade git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/huggingface_hub.git

# 8. Optional: (recommended) install HF CLI if you want to pre-download GGUF files
pip install --upgrade huggingface_hub

# 9. Optional: pre-download GGUF (uncomment to use; faster / avoids hub throttling)
# if [ -n "$HF_TOKEN" ]; then
#   export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
# fi
# mkdir -p ~/models/glm-4.7-flash
# huggingface-cli download zai-org/GLM-4.7-Flash --pattern "*Q4_K_M*.gguf" --dir ~/models/glm-4.7-flash

# 10. Diagnostics (very useful)
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

# 11. Run vLLM server (trust remote code per Codecademy + use bfloat16 on H100)
#     - --trust-remote-code is required for some HF repos that ship custom code/configs
#     - --dtype bfloat16 is recommended on H100 for memory/perf; change to bf16/float16 as you prefer
stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
