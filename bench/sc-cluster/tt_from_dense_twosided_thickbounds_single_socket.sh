#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 60
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-15 ../../build/src/tt_from_dense_twosided_thickbounds_bench 2 30 $i 20
done
