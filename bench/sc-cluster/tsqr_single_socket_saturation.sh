#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 300
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for i in 1 5 10 25 50; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    srun likwid-pin -c 0-$t ../../build/src/tsqr_bench 10000000 $i 0 20
  done
done

for i in 100 250 500; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    srun likwid-pin -c 0-$t ../../build/src/tsqr_bench 500000 $i 0 20
  done
done
