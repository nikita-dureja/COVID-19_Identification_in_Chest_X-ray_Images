#!/bin/bash
#SBATCH --job-name=hypertuning
#SBATCH --output=hypertuning1.log
#SBATCH --error=hypertuning1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=skylake-gpu
#require one GPU
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00


module load anaconda3/5.1.0
module load openmpi/4.0.0
module load cudnn/7.6.5-cuda-10.1.243
source activate test

cd "/home/srenchi"

srun "/home/srenchi/.conda/envs/she/bin/python" "Hyper_tuning1.py"