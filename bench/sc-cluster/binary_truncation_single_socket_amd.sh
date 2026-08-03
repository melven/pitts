#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 600
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

totalSizeBytes=300000000 #0
totalSize=$(expr $totalSizeBytes / 8)

export OMP_WAIT_POLICY=ACTIVE
for i in 50 100 150 200 250 300 350 400 450 500; do
  n=$(expr $totalSize / $i)
  k=$(expr $i / 2)
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build_spack-26.1_/src/binary_truncation_bench $n $i $n $i 10 5 2
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build_spack-26.1_/src/binary_truncation_bench $n $i $n $k 10 5 2
done
