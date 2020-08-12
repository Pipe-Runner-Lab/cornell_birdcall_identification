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
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="debug_train_1.yml" --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="debug_train_2.yml" --dev -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_1.yml" --dev -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_2.yml" --dev -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_3.yml" --dev -p

# 2080Ti Basic run
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_4_a.yml" --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_4_b.yml" --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_4_c.yml" --dev
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_5_b.yml" --dev -p
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_5_c.yml" --dev -p

# Noise experiments
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_6_a.yml" --dev

# Pann networks
# CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_7_a.yml" --dev

# Enet larger batches
CUDA_VISIBLE_DEVICES=$GPU_IDX python3 main.py --config="exp_8_a.yml" --dev -h
