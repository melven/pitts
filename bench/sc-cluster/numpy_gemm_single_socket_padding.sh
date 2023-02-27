#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 600
#SBATCH --nodelist=be-cpu05
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

for ((i=2; i <= 100; i+=2)); do
  ihalf=$(expr $i / 2)
  # 2**24
  n=16777216
  echo "numpy_gemm with $n $i $ihalf 50"
  srun likwid-pin -c 0-15 python ../numpy_gemm_bench.py $n $i $ihalf 50
  echo "numpy_gemm with --padding $n $i $ihalf 50"
  srun likwid-pin -c 0-15 python ../numpy_gemm_bench.py --padding $n $i $ihalf 50
done
