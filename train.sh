#!/bin/bash

GPU_IDX=0

# ====================================================================================
#                                    CONFIG LIST
# ====================================================================================

# fold training
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_a.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_b.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_c.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_d.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_e.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_f.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1_g.yml" -p

# full data training
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_a.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_b.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_c.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_d.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_e.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_f.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2_g.yml" -p

# fulldata set cosine
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_a.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_b.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_c.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_d.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_e.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_f.yml" -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3_g.yml" -p

# Dummy run
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="debug_train_1.yml" --dev -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="debug_train_2.yml" --dev -p
CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1.yml" --dev
