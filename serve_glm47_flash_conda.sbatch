#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH --job-name=glm47_flash
#SBATCH -o glm47_flash.%j.out
#SBATCH -e glm47_flash.%j.err

############################################
# 1. Initialize Miniconda
############################################

# IMPORTANT: adjust this path if your miniconda is elsewhere
source ~/miniconda3/etc/profile.d/conda.sh

conda activate llm      # <-- your conda env name

############################################
# 2. CUDA / system modules
############################################

module load cuda/12.1   # adjust to your cluster
module load gcc/11      # often required by torch/nccl

############################################
# 3. Hugging Face cache (avoid HOME quota)
############################################

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p $HF_HOME

############################################
# 4. MLflow configuration
############################################

# MLflow tracking server (already running)
export MLFLOW_TRACKING_URI=http://compute-3:5000   # CHANGE THIS

export MODEL_NAME=GLM47-Flash
export MODEL_STAGE=Production

############################################
# 5. Sanity checks
############################################

echo "Node: $(hostname)"
echo "Conda env: $(which python)"

nvidia-smi

############################################
# 6. NCCL tuning (important on HPC)
############################################

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

############################################
# 7. Launch vLLM server
############################################

echo "Starting vLLM GLM-4.7-Flash server..."

vllm serve zai-org/GLM-4.7-Flash \
  --tensor-parallel-size 4 \
  --served-model-name glm-4.7-flash \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 \
  --port 8000
