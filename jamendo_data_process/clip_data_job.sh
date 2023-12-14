#!/bin/bash

#SBATCH --cpus-per-task=20
#SBATCH -o process_jamendo_data.txt
#SBATCH --job-name=process_jamendo_data

module load anaconda/2023a

python clip_data.py
