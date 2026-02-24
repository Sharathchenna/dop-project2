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

# 1. Init environment
source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm
export PYTHONUNBUFFERED=1

# 2. Cache
export HF_HOME=$HOME/hf_cache
mkdir -p $HF_HOME

# 3. Dependency installs
echo "🔍 Installing dependencies..."

# ---- PyTorch CUDA 12.1
python - << 'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
EOF

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- Install latest Transformers from GitHub
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git

# ---- Install latest vLLM from GitHub
pip uninstall -y vllm
pip install git+https://github.com/vllm-project/vllm.git@main

# 4. Diagnostics
echo "=== ENVIRONMENT CHECK ==="
which python
which vllm
python - << 'EOF'
import torch, transformers, vllm
print("Torch CUDA:", torch.cuda.is_available())
print("Transformers version:", transformers.__version__)
print("vLLM version:", vllm.__version__)
EOF
nvidia-smi
echo "========================="

# 5. Launch model server
echo "🚀 Starting vLLM server..."
stdbuf -oL -eL vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
