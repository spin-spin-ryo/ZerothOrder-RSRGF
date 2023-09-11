#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o /home/u00786/Research/optimization/outputs/output.%j.out

export PATH=/home/app/singularity-ce/bin:$PATH
singularity exec --nv /home/u00786/Research/Master-Research1/pytorch_22.03-py3.sif python /home/u00786/Research/optimization/runNumerical.py