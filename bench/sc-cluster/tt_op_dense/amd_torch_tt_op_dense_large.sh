#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --threads-per-core=1
#SBATCH -p amd
#SBATCH --mem=200G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# no modules, miniforge3
ulimit -s unlimited
export OMP_STACKSIZE=100M
export MKL_NUM_THREADS=64

for((r=100;r<300;r++)); do
  echo "r: $r, nIter: 100"
  srun taskset -c 0-127 likwid-pin -c 0-63 ~/miniforge3/bin/python ../../torch_tt_op_dense_bench.py 50 $r 2 100
done

for((r=300;r<500;r++)); do
  echo "r: $r, nIter: 50"
  srun taskset -c 0-127 likwid-pin -c 0-63 ~/miniforge3/bin/python ../../torch_tt_op_dense_bench.py 50 $r 2 50
done

for((r=500;r<=700;r++)); do
  echo "r: $r, nIter: 25"
  srun taskset -c 0-127 likwid-pin -c 0-63 ~/miniforge3/bin/python ../../torch_tt_op_dense_bench.py 50 $r 2 25
done
