#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --nodelist=be-cpu05
#SBATCH --mem=90G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc12-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build_gcc_/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank  5 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 10 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 15 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 20 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 25 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 30 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 35 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 40 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 45 --preconditioner=TT-rank1
srun likwid-pin -c 0-15 python ttpy_amen.py -n 50 -d 10 --rhs_random_rank 50 --preconditioner=TT-rank1

