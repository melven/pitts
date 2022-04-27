#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1200
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc10-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited

srun likwid-pin -c 0-0 python tt_gmres_precond.py -n 20 -d 8 --eps 1.e-8 --maxIter=100  --preconditioner TT-rank1 --variant no_precond
