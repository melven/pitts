#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-13 ~/test/plasma-19.8.1/build/plasmatest dgeqrf --dim=10000000x$i --iter=10
done
