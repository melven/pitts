#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for i in 2 10 20 50 100; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    ihalf=$(expr $i / 2)
    srun likwid-pin -c 0-$t ../../build/src/tsmm_bench 10000000 $i $ihalf 50 5000000 $i
  done
done

for i in 200 500 1000; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    ihalf=$(expr $i / 2)
    srun likwid-pin -c 0-$t ../../build/src/tsmm_bench 500000 $i $ihalf 20 250000 $i
  done
done
