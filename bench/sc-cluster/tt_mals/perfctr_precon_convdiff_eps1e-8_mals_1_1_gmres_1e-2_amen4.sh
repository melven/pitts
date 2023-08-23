#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc12-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build_gcc_/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 20 -n 40 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 30 -n 60 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 40 -n 80 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 50 -n 100 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1

srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 6 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 8 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 12 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 14 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --preconditioner TT-rank1

srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --rhs_type rhs_random --rhs_random_rank 1 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --rhs_type rhs_random --rhs_random_rank 2 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --rhs_type rhs_random --rhs_random_rank 3 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --rhs_type rhs_random --rhs_random_rank 4 --preconditioner TT-rank1
srun likwid-perfctr -g FLOPS_DP -C 0-0 python tt_mals.py -L 1 -C 10 -n 20 -d 10 --eps 1.e-8 --nSweeps 20 --nMALS 1 --nAMEnEnrichment 4 --gmresRelTol 1.e-2 --rhs_type rhs_random --rhs_random_rank 5 --preconditioner TT-rank1
