#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 50; i++)); do
  likwid-pin -c 0-15 python ../tt_from_dense_dgesdd.py 2 27 $i 1
done
