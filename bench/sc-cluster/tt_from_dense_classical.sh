#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 120
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-15 ../../build/src/tt_from_dense_classical_bench 2 27 $i 1
done
