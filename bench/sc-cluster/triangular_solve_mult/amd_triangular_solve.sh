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

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 65536000    1    1 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 32768000    2    2 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 16384000    4    4 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  8192000    8    8 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  5461000   12   12 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  4096000   16   16 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  2048000   32   32 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  1536000   48   48 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  1024000   64   64 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   512000  128  128 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   256000  256  256 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   128000  512  512 100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench    64000 1024 1024 100

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 655360000    1    1 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 327680000    2    2 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench 163840000    4    4 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  81920000    8    8 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  54610000   12   12 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  40960000   16   16 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  20480000   32   32 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  15360000   48   48 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench  10240000   64   64 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   5120000  128  128 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   2560000  256  256 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench   1280000  512  512 10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/trsm_bench    640000 1024 1024 10
