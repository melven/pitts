#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 600
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 100; i++)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 2 30 $i 20 3
done
