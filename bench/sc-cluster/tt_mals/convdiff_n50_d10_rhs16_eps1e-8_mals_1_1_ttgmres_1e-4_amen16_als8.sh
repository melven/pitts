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

srun likwid-pin -c 0-0 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 16 --nonsimplifiedAMEn --gmresRelTol 1.e-4 --gmresMaxIter 250 --rhs_type rhs_random --rhs_random_rank 16 --AMEn_ALS_residualRank 8
