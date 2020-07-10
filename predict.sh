#!/bin/bash

GPU_IDX=0

# ====================================================================================
#                                    CONFIG LIST
# ====================================================================================

# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="test_4.yml" --vote
CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="test_1.yml" --vote --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="test_2.yml" --vote --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="test_3.yml" --vote --dev

