#!/bin/bash

#SBATCH -J udt
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o /data/danielsohn0827000/uni-dthon-project/log_test/%A-%x.out

source ~/.bashrc

# 2. 본인의 가상환경 활성화 (base 환경)
conda activate base
python3 test.py

# letting slurm know this code finished without any problem
exit 0
