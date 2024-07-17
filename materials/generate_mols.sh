#!/bin/bash
#SBATCH --job-name="gaudi"
#SBATCH --array 4-7
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

nvidia-smi
nvidia-smi -L

set -e  # fail fully on first line failure

module load Anaconda3/2022.10
module load CUDA/11.3.1 
conda activate $DATA/envs/GaUDI

python -u generation_guidance.py --arg_id $SLURM_ARRAY_TASK_ID --gen_run_name "rerun_diffseeds" --name_cond_predictor_run "noise_scaled_reg" --name_edm_run "edm_standard_params"
