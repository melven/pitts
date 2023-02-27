#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 60
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

totalSizeBytes=3000000000
totalSize=$(expr $totalSizeBytes / 8)

for i in 50 100 150 200 250 300 350 400 450 500; do
  n=$(expr $totalSize / $i)
  srun likwid-pin -c 0-15 ../../build/src/tsqr_bench $n $i 0 200
done
