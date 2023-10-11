#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 100; i++)); do
  srun likwid-pin -c 0-15 ../../build_gcc_new/src/tsqr_bench 25000000 $i 0 50
done
