#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --nodelist=be-cpu05
#SBATCH --mem=90G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc12-openmpi-python
ulimit -s unlimited
export OMP_STACKSIZE=100M

for((r=100;r<=700;r++)); do
  n=$((r * 50))
  m=$((r))
  k=$((r * 2))
  echo "n m k: $n $m $k"
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $n $m $k 100
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $k $m $n 100
  n=$((r * r))
  m=$((50 * 2))
  k=$((50 * 2))
  echo "n m k: $n $m $k"
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $n $m $k 100
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $k $m $n 100
  n=$((50 * r))
  m=$((r * 2))
  k=$((r))
  echo "n m k: $n $m $k"
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $n $m $k 100
  srun likwid-pin -c 0-15 python ../../numpy_gemm_bench.py $k $m $n 100
done
