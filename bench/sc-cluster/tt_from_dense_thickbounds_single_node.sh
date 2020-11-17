#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_thickbounds_bench 2 30 $i 20
done

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_thickbounds_bench 2 32 $i 20
done
