#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 300
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

totalSizeBytes=2400000000
totalSize=$(expr $totalSizeBytes / 8)

for m in 5 10 25 50 100 250 500; do
  n=$(expr $totalSize / $m)
  for ((r=7; r <= 50; r+=4)); do
    for ((bs=3; bs < 30; bs+=6)); do
      srun likwid-pin -c 0-15 ../../build/src/tsqr_bench $n $m $r 10 $bs
    done
  done
done
