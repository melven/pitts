#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 1 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 2 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 4 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 8 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 16 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 32 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 64 20
srun likwid-perfctr -m -g MEM_DP -C 0-13 -V 1 ../../build_likwid/src/tt_from_dense_autofusing_bench 2 30 128 20

