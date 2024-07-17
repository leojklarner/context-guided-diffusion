#!/bin/bash
#SBATCH --job-name="mood"
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G

nvidia-smi
nvidia-smi -L

set -e  # fail fully on first line failure

module load Anaconda3/2022.10
module load CUDA/11.3.1 
conda activate mood

python -u data/preprocess.py --dataset "ZINC250k"
python -u data/preprocess.py --dataset "ZINC500k"