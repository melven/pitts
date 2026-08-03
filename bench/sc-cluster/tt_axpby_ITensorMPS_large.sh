#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 1200
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

# everything installed in .julia
#export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=64

for r in 1 50 100 150 200 250 300 350 400 450 500; do
  srun taskset -c 0-127 julia ../tt_axpby_ITensorMPS.jl 5000 5 50 $r
done
