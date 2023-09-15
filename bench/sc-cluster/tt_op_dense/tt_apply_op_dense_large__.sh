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
  srun likwid-pin -c 0-15 ../../../build_gcc__/src/apply_op_dense_bench 50 $r 2 100
done
