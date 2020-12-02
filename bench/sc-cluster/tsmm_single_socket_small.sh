#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for ((i=2; i <= 50; i+=2)); do
  ihalf=$(expr $i / 2)
  srun likwid-pin -c 0-13 ../../build/src/tsmm_bench 10000000 $i $ihalf 100 5000000 $i
done
