#!/bin/bash

#SBATCH --time=20-00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vatrack
#SBATCH --mem=170000
#SBATCH --begin=now

module load cuda/11.1
module load anaconda

source activate vatrack

echo " Computing job " $SLURM_JOB_ID " on "$(hostname)

srun python test.py
