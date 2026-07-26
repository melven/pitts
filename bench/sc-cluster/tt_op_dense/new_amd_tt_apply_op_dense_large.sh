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

for r in 1 50 100 150 200 250 300 350 400 450 500; do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/apply_op_dense_bench 5000 $r 2 3
done
