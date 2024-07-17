#!/bin/bash

sbatch --account stat-cadd train_regressor_ours.sh
sbatch --account stat-cadd train_ensemble_weight_decay.sh
sbatch --account stat-cadd train_regressor_weight_decay.sh
sbatch --account stat-cadd train_regressor_ps.sh
sbatch --account stat-cadd train_pretrained_weight_decay.sh