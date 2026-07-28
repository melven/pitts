#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --threads-per-core=1
#SBATCH -p amd
#SBATCH --mem=200G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc12-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun taskset -c 0-127 likwid-pin -c 0 python tt_mals.py -L 1 -C 10 -n 1024 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 2 --AMEn_ALS_residualRank 4 --nonsimplifiedAMEn --gmresRelTol 1.e-4 --gmresMaxIter 250 --rhs_type rhs_ones --toQttOp
