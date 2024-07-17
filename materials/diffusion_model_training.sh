#!/bin/bash
#SBATCH --job-name="guido"
#SBATCH --array 0-1
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -e  # fail fully on first line failure

module load Anaconda3/2022.10
module load CUDA/11.3.1 
conda activate $DATA/envs/GaUDI

python -u train_edm.py --arg_id $SLURM_ARRAY_TASK_ID --name "edm_standard_params"
