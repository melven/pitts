#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

for i in 50 100 150 200 250 300 350 400 450 500; do
  #srun likwid-pin -c 0-13 ../../build/src/tsqr_bench 1000000 $i 10 20
  #srun likwid-pin -c 0-13 ../../build/src/tsqr_bench 1000000 $i 17 20
  #srun likwid-pin -c 0-13 ../../build/src/tsqr_bench 1000000 $i 27 20
  #srun likwid-pin -c 0-13 ../../build/src/tsqr_bench 1000000 $i 37 20
  srun likwid-pin -c 0-13 ../../build/src/tsqr_bench 1000000 $i 0 20
done
