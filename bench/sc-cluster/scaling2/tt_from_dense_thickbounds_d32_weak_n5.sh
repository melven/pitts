#!/bin/bash
#SBATCH --exclusive
#SBATCH -n 5
#SBATCH -c 14
#SBATCH --threads-per-core=1
#SBATCH -t 120
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

export OMP_NUM_THREADS=14

for i in 10 20 30 40 50; do
  srun --cpu-bind=core,verbose likwid-pin -c 0-13 ../../../build/src/tt_from_dense_thickbounds_bench 2 32 $i 10 2 1
done
