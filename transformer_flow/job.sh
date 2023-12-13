#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o generate_audio_print.txt
#SBATCH --job-name=generate_audio

module load anaconda/2023a

python test.py
