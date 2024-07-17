#!/bin/bash
#SBATCH --job-name="gaudi"
#SBATCH --array 4-7
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

nvidia-smi -L

set -e  # fail fully on first line failure

module load Anaconda3/2022.10
module load CUDA/11.3.1 
conda activate $DATA/envs/GaUDI

for i in {0..75}
do
    echo "Running job $i"
    python -u train_guidance_model.py --arg_id $SLURM_ARRAY_TASK_ID --hyper_id $i --name "noise_scaled_reg"
done
