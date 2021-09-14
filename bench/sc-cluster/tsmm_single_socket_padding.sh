#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for ((i=2; i <= 100; i+=2)); do
  ihalf=$(expr $i / 2)
  # 2**24
  n=16777216
  nhalf=$(expr $n / 2)
  srun likwid-pin -c 0-13 ../../build/src/tsmm_bench $n $i $ihalf 50 $nhalf $i
done
