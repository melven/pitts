#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --threads-per-core=1
#SBATCH -p amd
#SBATCH --mem=200G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc13-openmpi-python
ulimit -s unlimited
export OMP_STACKSIZE=100M

for((r=100;r<=700;r++)); do
  n=$((r * 50))
  m=$((r))
  k=$((r * 2))
  echo "n m k: $n $m $k"
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $n $m $k 100
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $k $m $n 100
  n=$((r * r))
  m=$((50 * 2))
  k=$((50 * 2))
  echo "n m k: $n $m $k"
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $n $m $k 100
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $k $m $n 100
  n=$((50 * r))
  m=$((r * 2))
  k=$((r))
  echo "n m k: $n $m $k"
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $n $m $k 100
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_gemm_bench.py $k $m $n 100
done
