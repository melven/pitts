#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_bench 2 30 $i 20
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_bench 4 15 $i 20
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_bench 8 10 $i 20
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_bench 10 9 $i 20
  srun likwid-pin -c 0-13 ../../build/src/tt_from_dense_bench 32 6 $i 3
done
