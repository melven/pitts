#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

for ((i=500; i <= 1500; i+=100)); do
  srun likwid-pin -c 0-15 ../../build_gcc_new/src/tsqr_bench 50000 $i 0 50
done
