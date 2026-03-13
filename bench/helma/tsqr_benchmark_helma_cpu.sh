#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -n 1
#SBATCH -c 384
#SBATCH --output="%x-%j.out"
#SBATCH --time=02:00:00


# source load_modules_pitts.sh
cd ../../build_new

likwid-pin -q -c 0-383 ./src/tsqr_double_bench 65536000    1  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 32768000    2  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 16384000    4  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  8192000    8  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  5461000   12  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  4096000   16  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  2048000   32  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  1536000   48  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  1024000   64  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   512000  128  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   256000  256 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   128000  512 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench    64000 1024 25 50

likwid-pin -q -c 0-383 ./src/tsqr_double_bench 655360000    1  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 327680000    2  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 163840000    4  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  81920000    8  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  54610000   12  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  40960000   16  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  20480000   32  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  15360000   48  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  10240000   64  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   5120000  128  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   2560000  256 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   1280000  512 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench    640000 1024 25 50

likwid-pin -q -c 0-383 ./src/tsqr_double_bench 6553600000    1  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 3276800000    2  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 1638400000    4  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  819200000    8  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  546100000   12  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  409600000   16  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  204800000   32  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  153600000   48  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  102400000   64  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   51200000  128  0 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   25600000  256 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   12800000  512 25 50
likwid-pin -q -c 0-383 ./src/tsqr_double_bench    6400000 1024 25 50

likwid-pin -q -c 0-383 ./src/tsqr_double_bench 65536000000    1  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 32768000000    2  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench 16384000000    4  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  8192000000    8  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  5461000000   12  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  4096000000   16  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  2048000000   32  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  1536000000   48  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench  1024000000   64  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   512000000  128  0 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   256000000  256 25 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench   128000000  512 25 5
likwid-pin -q -c 0-383 ./src/tsqr_double_bench    64000000 1024 25 5
