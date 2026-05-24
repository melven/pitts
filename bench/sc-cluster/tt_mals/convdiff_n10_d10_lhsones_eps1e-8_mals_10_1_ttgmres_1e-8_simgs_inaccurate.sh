#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 3000
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc12-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun likwid-pin -c 0-0 python tt_mals.py -L 1 -C 0.1 -n 10 -d 10 --eps 1.e-8 --nSweeps 1 --useTTgmres --nMALS 10 --gmresRelTol 1.e-8 --gmresMaxIter 100 --maxRank 500 --rhs_type lhs_ones
