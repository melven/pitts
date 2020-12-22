#!/bin/bash
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -c 1
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc10-openmpi-python

for ((i=1; i <= 50; i++)); do
  srun --cpu-bind=core ../trilinos/build/tsqr 10000000 $i 10
done
