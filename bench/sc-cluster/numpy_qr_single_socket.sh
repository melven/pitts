#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-13 python ../numpy_qr_bench.py 10000000 $i 10
done
