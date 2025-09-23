#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# everything installed in .julia

for ((i=1; i <= 50; i++)); do
  OMP_NUM_THREADS=16 julia ../tt_from_dense_ITensorMPS.jl 2 27 $i 1
done
