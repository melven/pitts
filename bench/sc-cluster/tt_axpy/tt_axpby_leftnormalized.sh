#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --nodelist=be-cpu05
#SBATCH --mem=90G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc13-openmpi-python
ulimit -s unlimited
export OMP_STACKSIZE=100M

for((r=1;r<=100;r++)); do
  srun likwid-pin -c 0-15 ../../../build_gcc_new/src/axpby_bench 50 10 50 $r 100 L L
done
