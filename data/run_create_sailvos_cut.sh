#!/bin/bash

#SBATCH --time=20-00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=0114
#SBATCH --mem=170000
#SBATCH --begin=now


module load cuda/11.1
source activate vatrack

echo " Computing job " $SLURM_JOB_ID " on "$(hostname)

srun python create_sailvoscut_fromdict.py
