#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 60
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

for i in 100 200 300 400 500 600 700 800 900 1000; do
  ihalf=$(expr $i / 2)
  srun likwid-pin -c 0-15 ../../build/src/tsmm_bench 1000000 $i $ihalf 50 500000 $i
done
