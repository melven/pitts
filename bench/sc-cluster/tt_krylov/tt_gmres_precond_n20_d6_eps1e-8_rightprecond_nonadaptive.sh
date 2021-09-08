#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 1200
#SBATCH --mem=350G
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc10-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun likwid-pin -c 0-13 python tt_gmres_precond.py -n 20 -d 6 --eps 1.e-8 --maxIter=90  --variant right_precond
