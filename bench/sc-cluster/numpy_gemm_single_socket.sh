#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for ((i=2; i <= 50; i+=2)); do
  ihalf=$(expr $i / 2)
  srun likwid-pin -c 0-13 python ../numpy_gemm_bench.py 10000000 $i $ihalf 100
done
