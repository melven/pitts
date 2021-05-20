#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc10-openmpi-python

for ((i=2; i <= 100; i+=2)); do
  ihalf=$(expr $i / 2)
  srun likwid-pin -c 0-13 ../../build/src/tsmm_bench 25000000 $i $ihalf 50 12500000 $i
done
