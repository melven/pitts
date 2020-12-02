#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 10
#SBATCH --exclusive

srun ~/likwid_install/bin/likwid-bench -s 10 -t peakflops_avx512_fma -w S0:210kB:14
