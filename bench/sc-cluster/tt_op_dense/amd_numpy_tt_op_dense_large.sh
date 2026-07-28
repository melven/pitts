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
# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate numpy
ulimit -s unlimited
export OMP_STACKSIZE=100M
export MKL_NUM_THREADS=64
export LD_PRELOAD="/home/zoel_ml/lib/libmkl_fakeintel.so"

for r in 1 50 100 150 200 250 300 350 400 450 500; do
  echo "r: $r"
  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../numpy_tt_op_dense_bench.py 5000 $r 2 1
done
