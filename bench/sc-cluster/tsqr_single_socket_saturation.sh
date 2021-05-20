#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 300
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load PrgEnv/gcc10-openmpi-python

totalSizeBytes=3000000000
totalSizePerCore=$(expr $totalSizeBytes / 8 / 14 )

for i in 1 5 10 25 50; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    tt=$(expr $t + 1)
    n=$(expr $totalSizePerCore / $i \* $tt )
    echo "n: $n, m: $i, threads: $tt"
    srun likwid-pin -c 0-$t ../../build/src/tsqr_bench $n $i 0 100
  done
done

for i in 100 250 500; do
  for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    tt=$(expr $t + 1)
    n=$(expr $totalSizePerCore / $i \* $tt )
    echo "n: $n, m: $i, threads: $tt"
    srun likwid-pin -c 0-$t ../../build/src/tsqr_bench $n $i 0 50
  done
done
