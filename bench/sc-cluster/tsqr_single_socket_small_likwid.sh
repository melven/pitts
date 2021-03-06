#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-perfctr -m -g MEM_DP -C 0-13 ../../build_likwid/src/tsqr_bench 10000000 $i 0 100
done
