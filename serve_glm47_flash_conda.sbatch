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
#   - Official HF model card: https://huggingface.co/zai-org/GLM-4.7-Flash
#   - Official vLLM recipes:  https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html
#   - vLLM tool calling docs: https://docs.vllm.ai/en/latest/features/tool_calling/
#
# Key facts:
#   --tool-call-parser glm47   is correct (confirmed by official docs)
#   --reasoning-parser glm45   is correct (confirmed by official docs)
#   glm47 parser was added recently — an old nightly wheel won't have it.
#   Fix: force install the LATEST nightly by explicitly no-caching pip.
#
# No git on cluster: use tarball URLs for GitHub-hosted packages.
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
#    If CUDA 12.4+, use --index-url https://download.pytorch.org/whl/cu124 instead
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Install the LATEST vLLM nightly.
#    --no-cache-dir forces pip to re-download rather than use a stale cached wheel.
#    This is critical — without it you may get an old nightly missing the glm47 parser.
pip install \
    --no-cache-dir \
    --upgrade \
    --pre \
    vllm \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# 6. Verify glm47 parser is present in the installed build
python - <<'PY'
import sys
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    parsers = list(ToolParserManager._tool_parsers.keys())
    print("Available tool parsers:", sorted(parsers))
    if "glm47" in parsers:
        print("✅ glm47 parser: FOUND")
    else:
        print("❌ glm47 parser: MISSING — nightly may still be too old")
        print("   Fallback: use --tool-call-parser glm45 temporarily")
except Exception as e:
    print("Parser check error:", e)
PY

# 7. Install transformers from GitHub tarball (no git required on cluster).
#    Must come AFTER vLLM so it overrides vLLM's pinned transformers version.
#    Using transformers 5.x (main) which accepts huggingface-hub >= 1.0,
#    resolving the huggingface-hub version conflict from transformers 4.57.x.
pip install --no-cache-dir --upgrade \
    "https://github.com/huggingface/transformers/archive/refs/heads/main.tar.gz"

# 8. Pin huggingface_hub to a version compatible with transformers 5.x
pip install --no-cache-dir "huggingface_hub>=1.0,<2.0"

# 9. Diagnostics
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
for pkg in ("torch", "transformers", "vllm", "huggingface_hub"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}:", getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(f"{pkg}: ERROR:", e)

# Check glm4_moe_lite arch is available (required for GLM-4.7-Flash)
try:
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
    print("glm4_moe_lite: ✅ OK")
except ImportError as e:
    print("glm4_moe_lite: ❌ MISSING —", e)
PY
nvidia-smi

# 10. Serve GLM-4.7-Flash
#
#    Flags confirmed against official zai-org HF model card and vLLM recipes docs:
#      --tool-call-parser glm47       official GLM-4.7 tool call parser
#      --reasoning-parser glm45       official GLM-4.x reasoning/thinking parser
#      --enable-auto-tool-choice      required to expose tool-calling via OpenAI API
#      --tokenizer-mode slow          fixes AttributeError: all_special_tokens_extended
#      --no-enable-prefix-caching     recommended by official vLLM GLM recipe
#      --disable-log-requests         reduces log noise
#      SAFETENSORS_FAST_GPU=1         recommended by official vLLM GLM recipe
#
#    NOTE: Thinking mode is ON by default. To disable per-request, pass:
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
