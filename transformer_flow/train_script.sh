#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o train_model_print_10layers_3e-3_mel.txt
#SBATCH --job-name=train_model

module load anaconda/2023a

python train.py --dataset square_mel_jamendo_data --epochs 1 --batch_size 5 --lr 3e-3 --num_layers 10 --dim_feedforward 128