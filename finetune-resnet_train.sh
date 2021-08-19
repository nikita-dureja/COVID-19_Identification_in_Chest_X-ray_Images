#!/bin/bash
#SBATCH --job-name=resnet_train
#SBATCH --output=finetune-resnet_final.log
#SBATCH --error=finetune-resnet_final.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=skylake-gpu
# require one GPU
#SBATCH --gres=gpu:1
#SBATCH --time=06:30:00

module load openmpi/4.0.0
module load cudnn/7.6.5-cuda-10.2.89
source activate /home/mrajopad/.conda/envs/test

cd "/home/mrajopad/"

srun "/home/mrajopad/.conda/envs/test/bin/python3" "finetune-resnet_train.py"
