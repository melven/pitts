#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH -t 3000
#SBATCH --output="%x-%j.out"


ulimit -s unlimited
export OMP_STACKSIZE=100M

for r in 1 50; do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/axpby_bench 5000 5 50 $r 100 L L
done

for r in 100 150 200 250; do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/axpby_bench 5000 5 50 $r 10 L L
done

for r in 300 350; do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/axpby_bench 5000 5 50 $r 5 L L
done

for r in 400 450 500; do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/axpby_bench 5000 5 50 $r 2 L L
done
