#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 6000
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"


for ((i=50; i <= 1000; i+=50)); do
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 2 30 $i 5
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 3 19 $i 5
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 4 15 $i 5
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 8 10 $i 5
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 10 9 $i 5
  srun taskset -c 0-127 likwid-pin -c 0-63 ../../build/src/tt_from_dense_thickbounds_bench 32 6 $i 3
done
