#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -c 1
#SBATCH -t 1200
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false

for ((i=1; i <= 100; i++)); do
  srun --cpu-bind=core ../trilinos/build/tsqr 25000000 $i 10
done
