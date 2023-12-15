#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o train_model_print_10layers_5e-4_mel.txt
#SBATCH --job-name=train_model

module load anaconda/2023a

python train.py --dataset square_mel_jamendo_data --epochs 10 --batch_size 5 --lr 5e-4 --num_layers 10 --dim_feedforward 128