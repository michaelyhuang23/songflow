#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o train_model_print_5layers_1e-5_old.txt
#SBATCH --job-name=train_model

module load anaconda/2023a

python train.py --dataset square_jamendo_data --epochs 1 --batch_size 5 --lr 1e-5 --num_layers 5 --dim_feedforward 128