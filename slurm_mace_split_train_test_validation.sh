#!/bin/bash
#SBATCH --job-name=mace
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -t 15-24:30


forces_weight=("10" "10")
stress_weight=("1" "5")
max_num_epochs=("40" "80")
len=${#forces_weight[@]}

export CUDA_VISIBLE_DEVICES=0
for ((i=0; i<len; i++)); do
    mace_run_train \
        --name="MACE_model" \
        --model="MACE" \
        --train_file="60.extxyz" \
        --valid_file="val.extxyz" \
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

