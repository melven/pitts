#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python
# # to avoid some delay through the intel omp library
# module unload intel-mkl

for ((i=1; i <= 100; i++)); do
  srun -n 1 -c 14 likwid-pin -c 0-13 ../../build/src/tsqr_bench 25000000 $i 0 50
done
