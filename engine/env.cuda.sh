#!/usr/bin/env sh

export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_PATH=/usr/local/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

