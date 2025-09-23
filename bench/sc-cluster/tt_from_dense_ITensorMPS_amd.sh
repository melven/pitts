#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 1200
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

# everything installed in .julia

for ((i=30; i <= 50; i++)); do
  srun taskset -c 0-127 julia ../tt_from_dense_ITensorMPS.jl 2 30 $i 1
done
