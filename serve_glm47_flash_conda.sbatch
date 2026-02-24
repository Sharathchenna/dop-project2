#!/bin/bash
#SBATCH -p gpu_h100_4              # correct partition
#SBATCH -N 1
#SBATCH --gres=gpu:1               # ✅ ONLY 1× H100
#SBATCH --cpus-per-task=4          # minimal CPU (vLLM is GPU-bound)
#SBATCH --mem=200G                 # safe for 1 GPU
#SBATCH -t 24:00:00
#SBATCH --job-name=glm47_flash_1gpu
#SBATCH -o glm47_flash_1gpu.%j.out
#SBATCH -e glm47_flash_1gpu.%j.err

############################################
# 1. Initialize Miniconda
############################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

############################################
# 2. System modules
############################################

module load cuda/12.1
module load gcc/11

############################################
# 3. Hugging Face cache (SCRATCH)
############################################

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p $HF_HOME

############################################
# 4. Diagnostics
############################################

echo "Node: $(hostname)"
which python
nvidia-smi

############################################
# 5. NCCL / CUDA safety
############################################

export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

############################################
# 6. Launch vLLM (single GPU)
############################################

vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 1 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 \
  --port 8000
