#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o train_model_print_10layers_3e-6.txt
#SBATCH --job-name=train_model

module load anaconda/2023a

python train.py --dataset square_jamendo_data --epochs 2 --batch_size 5 --lr 3e-6 --num_layers 10 --dim_feedforward 128