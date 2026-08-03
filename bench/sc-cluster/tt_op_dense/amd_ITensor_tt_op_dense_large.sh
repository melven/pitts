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
export OPENBLAS_NUM_THREADS=64

for r in 1 50 100 150 200 250 300 350 400 450 500; do
  echo "r: $r"
  srun taskset -c 0-127 julia ../../tt_op_dense_ITensorMPS.jl 5000 $r 2
done
