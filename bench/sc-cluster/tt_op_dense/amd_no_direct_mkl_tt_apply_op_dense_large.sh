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
#ulimit -s unlimited
#export OMP_STACKSIZE=100M

for((r=100;r<300;r++)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_gcc_amd_no_direct_mkl/src/apply_op_dense_bench 50 $r 2 100
done

for((r=300;r<500;r++)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_gcc_amd_no_direct_mkl/src/apply_op_dense_bench 50 $r 2 50
done

for((r=500;r<=700;r++)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_gcc_amd_no_direct_mkl/src/apply_op_dense_bench 50 $r 2 25
done
