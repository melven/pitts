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

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 32768000    2 16384000    4    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 16384000    4  8192000    8    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  8192000    8  4096000   16    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  4096000   16  2048000   32    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  2048000   32  1024000   64    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  1024000   64   512000  128    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   512000  128   256000  256    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   256000  256   128000  512    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   128000  512    64000 1024    0    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench    64000 1024    32000 2048    0    100

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 32768000    2 16384000    4    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 16384000    4  8192000    8    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  8192000    8  4096000   16    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  4096000   16  2048000   32    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  2048000   32  1024000   64    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  1024000   64   512000  128    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   512000  128   256000  256    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   256000  256   128000  512    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   128000  512    64000 1024    1    100
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench    64000 1024    32000 2048    1    100


srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 327680000    2 163840000    4    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 163840000    4  81920000    8    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  81920000    8  40960000   16    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  40960000   16  20480000   32    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  20480000   32  10240000   64    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  10240000   64   5120000  128    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   5120000  128   2560000  256    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   2560000  256   1280000  512    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   1280000  512    640000 1024    0    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench    640000 1024    320000 2048    0    10

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 327680000    2 163840000    4    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench 163840000    4  81920000    8    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  81920000    8  40960000   16    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  40960000   16  20480000   32    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  20480000   32  10240000   64    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench  10240000   64   5120000  128    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   5120000  128   2560000  256    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   2560000  256   1280000  512    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench   1280000  512    640000 1024    1    10
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/transpose_bench    640000 1024    320000 2048    1    10

