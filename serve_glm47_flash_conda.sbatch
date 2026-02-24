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
# - Uses GitHub tarball URLs instead of git+https:// because git is not
#   available on this HPC cluster.
# - vLLM nightly is required; glm4_moe_lite arch is not in stable releases.
# - transformers must be upgraded to 5.x AFTER vLLM install to resolve the
#   huggingface-hub<1.0 vs >=1.0 conflict (vLLM nightly pins 4.57.x,
#   but 5.x accepts huggingface-hub 1.x).
# - --tokenizer-mode slow fixes AttributeError: all_special_tokens_extended.
# - --tool-call-parser glm47 is correct for GLM-4.7 family (not glm45).
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

# 4. Install PyTorch for CUDA 12.1 (adjust index URL if cluster uses 12.4+)
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Install vLLM nightly — required for glm4_moe_lite architecture support.
#    Do NOT add --no-deps; let vLLM resolve its own pinned deps first.
#    The nightly wheel index is what the official HF model card specifies.
pip install -U vllm \
    --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# 6. Upgrade transformers to 5.x using the GitHub tarball (no git required).
#    This resolves the huggingface-hub version conflict:
#      transformers 4.57.x (installed by vLLM) requires huggingface-hub<1.0
#      transformers 5.x requires huggingface-hub>=1.0  <-- compatible
#    Must come AFTER vLLM so it overrides vLLM's pinned 4.57.x.
pip install --upgrade \
    "https://github.com/huggingface/transformers/archive/refs/heads/main.tar.gz"

# 7. Ensure huggingface_hub is at a compatible version for transformers 5.x
pip install "huggingface_hub>=1.0"

# 8. Diagnostics
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED or import error:", e)

# Sanity-check that glm4_moe_lite is present in this transformers build
try:
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
    print("glm4_moe_lite: OK (Glm4MoeLiteConfig found)")
except ImportError as e:
    print("glm4_moe_lite: MISSING —", e)

# Sanity-check the tokenizer attribute that was previously failing
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    _ = tok.all_special_tokens_extended
    print("all_special_tokens_extended: OK")
except Exception as e:
    print("all_special_tokens_extended check:", e)
PY
nvidia-smi

# 9. Serve GLM-4.7-Flash
#    Key flags:
#      --tokenizer-mode slow          fixes AttributeError: all_special_tokens_extended
#      --tool-call-parser glm47       correct parser for GLM-4.7 family (glm47, NOT glm45)
#      --reasoning-parser glm45       correct thinking/reasoning parser (glm45 for all GLM-4.x)
#      --enable-auto-tool-choice      exposes tool-calling via OpenAI-compatible API
#      --no-enable-prefix-caching     recommended by official vLLM GLM recipe
#      --disable-log-requests         reduces log noise
#      SAFETENSORS_FAST_GPU=1         recommended by official vLLM GLM recipe
#
#    NOTE: Thinking mode is ON by default. To disable it in a request, pass:
#      extra_body={"chat_template_kwargs": {"enable_thinking": false}}

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
