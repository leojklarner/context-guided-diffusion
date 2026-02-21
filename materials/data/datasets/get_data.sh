#!/bin/bash
#SBATCH --job-name="dowload_data"
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

wget https://zenodo.org/record/7798697/files/PASs_csv.tar.gz?download=1 -O PASs_csv.tar.gz
wget https://zenodo.org/record/7798697/files/PASs_xyz.tar.gz?download=1 -O PASs_xyz.tar.gz
wget https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/COMPAS-1x-xyzs.tar.gz?ref_type=heads -O COMPAS-1x-xyzs.tar.gz
wget https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/COMPAS-1x.csv? -O COMPAS-1x.csv

tar -xzf COMPAS-1x-xyzs.tar.gz
tar -xf PASs_xyz.tar.gz
