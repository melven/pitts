#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -n 384
#SBATCH -c 1
#SBATCH --output="%x-%j.out"
#SBATCH --time=05:00:00


# spack load some trilinos
cd ../trilinos/build
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
#export I_MPI_DEBUG=5

mpirun -np 384 ./tsqr 65536000    1 10
mpirun -np 384 ./tsqr 32768000    2 10
mpirun -np 384 ./tsqr 16384000    4 10
mpirun -np 384 ./tsqr  8192000    8 10
mpirun -np 384 ./tsqr  5461000   12 10
mpirun -np 384 ./tsqr  4096000   16 10
mpirun -np 384 ./tsqr  2048000   32 10
mpirun -np 384 ./tsqr  1536000   48 10
mpirun -np 384 ./tsqr  1024000   64 10
mpirun -np 384 ./tsqr   512000  128 10
mpirun -np 384 ./tsqr   256000  256 10
mpirun -np 384 ./tsqr   128000  512 10
mpirun -np 384 ./tsqr    64000 1024 10

mpirun -np 384 ./tsqr 655360000    1 10
mpirun -np 384 ./tsqr 327680000    2 10
mpirun -np 384 ./tsqr 163840000    4 10
mpirun -np 384 ./tsqr  81920000    8 10
mpirun -np 384 ./tsqr  54610000   12 10
mpirun -np 384 ./tsqr  40960000   16 10
mpirun -np 384 ./tsqr  20480000   32 10
mpirun -np 384 ./tsqr  15360000   48 10
mpirun -np 384 ./tsqr  10240000   64 10
mpirun -np 384 ./tsqr   5120000  128 10
mpirun -np 384 ./tsqr   2560000  256 10
mpirun -np 384 ./tsqr   1280000  512 10
mpirun -np 384 ./tsqr    640000 1024 10

mpirun -np 384 ./tsqr 6553600000    1 2
mpirun -np 384 ./tsqr 3276800000    2 2
mpirun -np 384 ./tsqr 1638400000    4 2
mpirun -np 384 ./tsqr  819200000    8 2
mpirun -np 384 ./tsqr  546100000   12 2
mpirun -np 384 ./tsqr  409600000   16 2
mpirun -np 384 ./tsqr  204800000   32 2
mpirun -np 384 ./tsqr  153600000   48 2
mpirun -np 384 ./tsqr  102400000   64 2
mpirun -np 384 ./tsqr   51200000  128 2
mpirun -np 384 ./tsqr   25600000  256 2
mpirun -np 384 ./tsqr   12800000  512 2
mpirun -np 384 ./tsqr    6400000 1024 2
