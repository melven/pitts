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

srun likwid-pin -c 0-15 python ttpy_amen.py

