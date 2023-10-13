#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --threads-per-core=1
#SBATCH -p amd
#SBATCH --mem=200G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


# module load PrgEnv/gcc13-openmpi-python
export PYTHONPATH=$PYTHONPATH:~/pitts/build_gcc_amd_slowcontract/src/:~/pitts/examples/
export PYTHONUNBUFFERED=1
ulimit -s unlimited
export OMP_STACKSIZE=100M

srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment  5 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 5  --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 10 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 10 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 15 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 15 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 20 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 20 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 25 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 25 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 30 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 30 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 35 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 35 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 40 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 40 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 45 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 45 --maxRank 500 --nonsimplifiedAMEn
srun taskset -c 0-127 likwid-pin -c 0-63 python tt_mals.py -L 1 -C 10 -n 50 -d 10 --eps 1.e-8 --nSweeps 200 --nMALS 1 --nAMEnEnrichment 50 --gmresRelTol 1.e-8 --rhs_type rhs_random --rhs_random_rank 50 --maxRank 500 --nonsimplifiedAMEn

