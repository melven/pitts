#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 600
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

# module load spack-user
# module load py-opt-einsum
export PYTHONPATH=~/teneva:$PYTHONPATH

for ((i=1; i <= 50; i++)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../tt_from_dense_teneva.py 2 30 $i 1
done
