#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# module load spack-user
# module load miniforge3
# eval "$(conda shell.bash hook)"
# conda activate /home/zoel_ml/conda_pytorch_env

for ((i=1; i <= 50; i++)); do
  python ../tt_from_dense_torchtt.py 2 27 $i 1
done
