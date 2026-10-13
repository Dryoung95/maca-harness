#!/usr/bin/env bash
# C500 + KernelBench environment — source this before any kernelbench run
# Resolve this script's own directory so the harness works regardless of where
# the repo is checked out (previous version hardcoded /data/cuda-harness-migration).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MACA_PATH=/opt/maca-3.3.0
export CUDA_HOME=$MACA_PATH/tools/cu-bridge
export PATH=$MACA_PATH/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib64:$LD_LIBRARY_PATH
# openai/litellm are unavailable offline and only serve the LLM-generation path;
# shims/ provides import stubs so kernelbench's top-level imports resolve.
export PYTHONPATH=$REPO_DIR/shims:$REPO_DIR/KernelBench/src
# 用户必须在 video 组才能访问 /dev/mxcd
