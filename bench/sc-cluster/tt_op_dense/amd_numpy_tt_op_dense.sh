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

for((r=10;r<=100;r+=10)); do
  echo "r: $r"
  srun taskset -c 0-127 likwid-pin -c 0-63 ~/miniforge3/bin/python ../../numpy_tt_op_dense_bench.py 50 $r 2 10
  #srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_tt_op_dense_bench.py 50 $r 2 10
done
