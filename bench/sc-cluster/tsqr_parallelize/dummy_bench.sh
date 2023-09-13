#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 30
#SBATCH --mem=10G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"
#SBATCH --exclusive

export OMP_NUM_THREADS=64
srun likwid-pin -c 0-63 ../../../build/src/tsqr_parallel > results_dummy_bench_node.txt

export OMP_NUM_THREADS=16
srun likwid-pin -c 0-15 ../../../build/src/tsqr_parallel > results_dummy_bench_cpu.txt

