#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 1200
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

for ((i=1; i <= 50; i++)); do
  srun likwid-pin -c 0-15 python ../numpy_qr_bench.py 25000000 $i 10
done
