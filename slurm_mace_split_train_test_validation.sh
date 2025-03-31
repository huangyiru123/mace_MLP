#!/bin/bash
#SBATCH --job-name=MACE_model_job         # 作业名称
#SBATCH --output=MACE_model_output.log    # 标准输出文件
#SBATCH --error=MACE_model_error.log      # 错误输出文件
#SBATCH --time=infinite                   # 预计运行时间
#SBATCH --mem=32GB                        # 申请的内存
#SBATCH --gres=gpu:1                      # 请求一个 GPU
#SBATCH --cpus-per-task=32                 # 每个任务分配的 CPU 核心数

# 加载必要的模块
module load cuda/11.3
module load python/3.8


forces_weight=("10" "10")
stress_weight=("1" "5")
max_num_epochs=("40" "80")
len=${#forces_weight[@]}

export CUDA_VISIBLE_DEVICES=0
for ((i=0; i<len; i++)); do
    mace_run_train \
        --name="MACE_model" \
        --model="MACE" \
        --train_file="train.extxyz" \
        --valid_file="valid.extxyz" \
        --test_file="test.extxyz" \
        --E0s="average" \
        --loss='universal' \
        --energy_weight=1 \
        --forces_weight=${forces_weight[i]} \
        --compute_stress=True \
        --stress_weight=${stress_weight[i]} \
        --stress_key='stress' \
        --energy_key='energy' \
        --forces_key='forces' \
        --eval_interval=1 \
        --error_table='PerAtomMAE' \
        --model="MACE" \
        --interaction_first="RealAgnosticDensityInteractionBlock" \
        --interaction="RealAgnosticDensityResidualInteractionBlock" \
        --num_interactions=2 \
        --correlation=3 \
        --max_ell=3 \
        --r_max=5.0 \
        --max_L=1 \
        --num_channels=128 \
        --num_radial_basis=8 \
        --MLP_irreps="16x0e" \
        --scaling='rms_forces_scaling' \
        --lr=0.008 \
        --weight_decay=1e-8 \
        --ema \
        --ema_decay=0.995 \
        --scheduler_patience=5 \
        --batch_size=6 \
        --valid_batch_size=6 \
        --pair_repulsion \
        --distance_transform="Agnesi" \
        --max_num_epochs=${max_num_epochs[i]} \
        --patience=40 \
        --amsgrad \
        --device=cuda \
        --seed=3 \
        --clip_grad=100 \
        --restart_latest \
        --default_dtype="float64"
done
