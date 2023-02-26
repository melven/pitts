#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 60
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"


for ((i=2; i <= 100; i+=2)); do
  ihalf=$(expr $i / 2)
  srun likwid-pin -c 0-15 ../../build/src/tsmm_bench 25000000 $i $ihalf 50 12500000 $i
done
