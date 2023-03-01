#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 300
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-15 ../../build/src/tt_from_dense_thickbounds_bench 2 30 $i 20
done
