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

totalSizeBytes=3000000000
totalSize=$(expr $totalSizeBytes / 8)

for i in 50 100 150 200 250 300 350 400 450 500; do
  n=$(expr $totalSize / $i)
  srun likwid-pin -c 0-13 ../../build/src/tsqr_bench $n $i 0 200
done
