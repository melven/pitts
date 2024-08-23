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

for((r=1;r<=100;r++)); do
  echo "r: $r"
  srun taskset -c 0-127 likwid-pin -c 0-63 ~/miniforge3/bin/python ../../torch_tt_op_dense_bench.py 50 $r 2 1000
  #srun taskset -c 0-127 likwid-pin -c 0-63 python ../../torch_tt_op_dense_bench.py 50 $r 2 1000
  #srun taskset -c 0-127 python ../../torch_tt_op_dense_bench.py 50 $r 2 1000
done
