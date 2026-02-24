#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --job-name=glm47_flash_1gpu
#SBATCH -o glm47_flash_1gpu.%j.out
#SBATCH -e glm47_flash_1gpu.%j.err

############################################
# 1. Environment init
############################################

source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONUNBUFFERED=1

############################################
# 2. Hugging Face cache
############################################

export HF_HOME=$HOME/hf_cache
mkdir -p $HF_HOME

############################################
# 3. Dependency checks (SAFE)
############################################

echo "🔍 Checking PyTorch + CUDA..."
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

echo "🔍 Checking vLLM..."
if ! command -v vllm &> /dev/null; then
    echo "📦 Installing vLLM"
    pip install vllm
fi

echo "🔍 Checking Transformers version..."
TRANSFORMERS_OK=0
python - << 'EOF'
import transformers
print("Transformers version:", transformers.__version__)
if "dev" in transformers.__version__:
    exit(0)
exit(1)
EOF
TRANSFORMERS_OK=$?

if [ $TRANSFORMERS_OK -ne 0 ]; then
    echo "📦 Installing Transformers (GitHub dev for GLM-4)"
    pip uninstall -y transformers
    pip install git+https://github.com/huggingface/transformers.git
else
    echo "✅ Transformers dev version already installed"
fi

############################################
# 4. Final diagnostics
############################################

echo "================ FINAL CHECK ================"
which python
which vllm
python - << 'EOF'
import torch, transformers
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
EOF
nvidia-smi
echo "============================================"

############################################
# 5. Start vLLM server
############################################

echo "🚀 Starting vLLM server..."

stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
