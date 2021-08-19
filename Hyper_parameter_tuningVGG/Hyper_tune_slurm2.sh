#!/bin/bash
#SBATCH --job-name=hyper_tuning
#SBATCH --output=hyper_tuning2.log
#SBATCH --error=hyper_tuning2.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=skylake-gpu
#require one GPU
#SBATCH --gres=gpu:1
#SBATCH --time=4:30:00


module load anaconda3/5.1.0
module load openmpi/4.0.0
module load cudnn/7.6.5-cuda-10.1.243
source activate test

cd "/fred/oz138/COS80024/project1"

srun "/home/srenchi/.conda/envs/test/bin/python" "HyperTuning2.py"