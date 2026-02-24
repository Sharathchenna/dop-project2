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
# 1. Proper environment initialization
############################################

source /etc/profile
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

############################################
# 2. Hugging Face cache (HOME-safe)
############################################

export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

############################################
# 3. Install dependencies IF missing
############################################

echo "Checking vLLM installation..."

if ! command -v vllm &> /dev/null; then
    echo "vLLM not found — installing dependencies"

    pip install --upgrade pip

    # Install PyTorch for CUDA 12.1 (H100 compatible)
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    # Install vLLM
    pip install vllm
else
    echo "vLLM already installed"
fi

############################################
# 4. Diagnostics
############################################

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "vLLM: $(which vllm)"

nvidia-smi

############################################
# 5. Launch vLLM (1× H100)
############################################

vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
