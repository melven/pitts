#!/bin/bash
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -c 14
#SBATCH --threads-per-core=1
#SBATCH -t 120
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

export OMP_NUM_THREADS=14

for ((i=1; i <= 50; i++)); do
  srun -N 4 -n 16 --threads-per-core=1 --cpu-bind=core,verbose ../../../build/src/tt_from_dense_thickbounds_bench 2 34 $i 40
done
