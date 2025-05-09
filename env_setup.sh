#!/bin/bash

# Activate conda or virtualenv if needed here
# source activate llama_env

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0,1

# For bfloat16 support if needed
export BNB_CUDA_VERSION=122

# Optional: Reduce tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Print environment setup
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BNB_CUDA_VERSION=$BNB_CUDA_VERSION"