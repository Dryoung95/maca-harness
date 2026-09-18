#!/usr/bin/env bash
# C500 + KernelBench environment — source this before any kernelbench run
export MACA_PATH=/opt/maca-3.3.0
export CUDA_HOME=$MACA_PATH/tools/cu-bridge
export PATH=$MACA_PATH/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/data/cuda-harness-migration/KernelBench/src
# coder 用户必须在 video 组才能访问 /dev/mxcd（已由 sudo usermod -aG video coder 配置）
