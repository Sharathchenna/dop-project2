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
# 1. Initialize environment (REQUIRED)
############################################

set -e

source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONUNBUFFERED=1

############################################
# 2. Hugging Face cache (safe location)
############################################

export HF_HOME=$HOME/hf_cache
mkdir -p $HF_HOME

############################################
# 3. Install dependencies (only if missing)
############################################

echo "🔍 Checking dependencies..."

# ---- PyTorch (CUDA 12.x, H100 compatible)
if ! python - << 'EOF'
import torch
print(torch.cuda.is_available())
EOF
then
    echo "📦 Installing PyTorch (CUDA 12.x)"
    pip install --upgrade pip
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
fi

# ---- vLLM
if ! command -v vllm &> /dev/null; then
    echo "📦 Installing vLLM"
    pip install vllm
fi

# ---- Transformers (GLM-4 requires dev version)
python - << 'EOF'
import transformers
exit(0 if "dev" in transformers.__version__ else 1)
EOF

if [ $? -ne 0 ]; then
    echo "📦 Installing Transformers (GitHub dev version for GLM-4)"
    pip uninstall -y transformers
    pip install git+https://github.com/huggingface/transformers.git
fi

############################################
# 4. Diagnostics (VERY IMPORTANT)
############################################

echo "================ ENV CHECK ================"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "vLLM: $(which vllm)"
python - << 'EOF'
import torch, transformers
print("Torch CUDA available:", torch.cuda.is_available())
print("Transformers version:", transformers.__version__)
EOF
nvidia-smi
echo "==========================================="

############################################
# 5. Launch vLLM server (1× H100)
############################################

echo "🚀 Starting vLLM server..."

stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
