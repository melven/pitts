#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# module load spack-user
# module load py-opt-einsum
export PYTHONPATH=~/teneva:$PYTHONPATH

for ((i=1; i <= 50; i++)); do
  likwid-pin -c 0-15 python ../tt_from_dense_teneva.py 2 27 $i 1
done
