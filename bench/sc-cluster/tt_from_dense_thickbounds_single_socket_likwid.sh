#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 300
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-perfctr -g MEM_DP -C 0-13 -m ../../build_likwid/src/tt_from_dense_thickbounds_bench 2 30 $i 20
done
