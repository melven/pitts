#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 60
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

totalSizeBytes=2400000000
totalSize=$(expr $totalSizeBytes / 8)

for m in 5 10 25 50 100 250 500; do
  n=$(expr $totalSize / $m)
  for ((r=7; r <= 50; r+=4)); do
    for ((bs=3; bs < 30; bs+=6)); do
      srun -n 1 -c 14 likwid-pin -c 0-13 ../../build/src/tsqr_bench $n $m $r 10 $bs
    done
  done
done
