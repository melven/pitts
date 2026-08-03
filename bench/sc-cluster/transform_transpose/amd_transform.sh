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

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench 32768000    2    1 100 16384000    2
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench 16384000    4    2 100  8192000    4
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  8192000    8    4 100  4096000    8
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  4096000   16    8 100  2048000   16
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  2048000   32   16 100  1024000   32
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  1024000   64   32 100   512000   64
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   512000  128   64 100   256000  128
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   256000  256  128 100   128000  256
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   128000  512  256 100    64000  512
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench    64000 1024  512 100    32000 1024
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench    32000 2048 1024 100    16000 2048

srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench 327680000    2    1 10 163840000    2
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench 163840000    4    2 10  81920000    4
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  81920000    8    4 10  40960000    8
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  40960000   16    8 10  20480000   16
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  20480000   32   16 10  10240000   32
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench  10240000   64   32 10   5120000   64
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   5120000  128   64 10   2560000  128
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   2560000  256  128 10   1280000  256
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench   1280000  512  256 10    640000  512
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench    640000 1024  512 10    320000 1024
srun taskset -c 0-127 likwid-pin -c 0-63 ../../../build_spack-26.1/src/tsmm_bench    320000 2048 1024 10    160000 2048

