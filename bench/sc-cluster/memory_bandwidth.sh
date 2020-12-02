#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 10
#SBATCH --exclusive

# module load PrgEnv/gcc10-openmpi-python

srun likwid-bench -s 10 -t load_avx512 -w S0:1GB:14
srun likwid-bench -s 10 -t copy_mem_avx512 -w S0:1GB:14
srun likwid-bench -s 10 -t stream_mem_avx512 -w S0:1GB:14
srun likwid-bench -s 10 -t store_mem_avx512 -w S0:1GB:14
