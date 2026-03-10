#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 100; i++)); do
  likwid-pin -c 0-15 ../../build_intel/src/tt_from_dense_bench 2 30 $i 20
done
