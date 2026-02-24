#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --job-name=glm47_flash
#SBATCH -o glm47_flash.%j.out
#SBATCH -e glm47_flash.%j.err

set -e
set -x

# -------------------------------
# Environment
# -------------------------------
source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm
export PYTHONUNBUFFERED=1

# -------------------------------
# HuggingFace cache (NO /hf_cache)
# -------------------------------
export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME

# Optional but recommended
# export HF_TOKEN=xxxxxxxxxxxxxxxx

# -------------------------------
# Clean broken installs
# -------------------------------
pip uninstall -y vllm transformers torch

# -------------------------------
# PyTorch (CUDA 12.x compatible)
# -------------------------------
pip install --upgrade pip
pip install torch==2.4.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# -------------------------------
# Transformers (GitHub main)
# -------------------------------
pip install git+https://github.com/huggingface/transformers.git

# -------------------------------
# vLLM (PREBUILT – NO COMPILATION)
# -------------------------------
pip install vllm==0.11.2

# -------------------------------
# Sanity checks
# -------------------------------
python - << 'EOF'
import torch, transformers, vllm
print("CUDA:", torch.cuda.is_available())
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
EOF

nvidia-smi

# -------------------------------
# Start server
# -------------------------------
echo "🚀 Starting vLLM server..."

stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
