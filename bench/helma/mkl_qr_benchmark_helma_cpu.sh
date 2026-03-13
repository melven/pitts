#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -n 1
#SBATCH -c 384
#SBATCH --output="%x-%j.out"
#SBATCH --time=05:00:00


# source load_modules_pitts.sh
cd ../../build_new

likwid-pin -q -c 0-383 ./src/qr_bench 65536000    1 10
likwid-pin -q -c 0-383 ./src/qr_bench 32768000    2 10
likwid-pin -q -c 0-383 ./src/qr_bench 16384000    4 10
likwid-pin -q -c 0-383 ./src/qr_bench  8192000    8 10
likwid-pin -q -c 0-383 ./src/qr_bench  5461000   12 10
likwid-pin -q -c 0-383 ./src/qr_bench  4096000   16 10
likwid-pin -q -c 0-383 ./src/qr_bench  2048000   32 10
likwid-pin -q -c 0-383 ./src/qr_bench  1536000   48 10
likwid-pin -q -c 0-383 ./src/qr_bench  1024000   64 10
likwid-pin -q -c 0-383 ./src/qr_bench   512000  128 10
likwid-pin -q -c 0-383 ./src/qr_bench   256000  256 10
likwid-pin -q -c 0-383 ./src/qr_bench   128000  512 10
likwid-pin -q -c 0-383 ./src/qr_bench    64000 1024 10

likwid-pin -q -c 0-383 ./src/qr_bench 655360000    1 10
likwid-pin -q -c 0-383 ./src/qr_bench 327680000    2 10
likwid-pin -q -c 0-383 ./src/qr_bench 163840000    4 10
likwid-pin -q -c 0-383 ./src/qr_bench  81920000    8 10
likwid-pin -q -c 0-383 ./src/qr_bench  54610000   12 10
likwid-pin -q -c 0-383 ./src/qr_bench  40960000   16 10
likwid-pin -q -c 0-383 ./src/qr_bench  20480000   32 10
likwid-pin -q -c 0-383 ./src/qr_bench  15360000   48 10
likwid-pin -q -c 0-383 ./src/qr_bench  10240000   64 10
likwid-pin -q -c 0-383 ./src/qr_bench   5120000  128 10
likwid-pin -q -c 0-383 ./src/qr_bench   2560000  256 10
likwid-pin -q -c 0-383 ./src/qr_bench   1280000  512 10
likwid-pin -q -c 0-383 ./src/qr_bench    640000 1024 10

likwid-pin -q -c 0-383 ./src/qr_bench 6553600000    1 2
likwid-pin -q -c 0-383 ./src/qr_bench 3276800000    2 2
likwid-pin -q -c 0-383 ./src/qr_bench 1638400000    4 2
likwid-pin -q -c 0-383 ./src/qr_bench  819200000    8 2
likwid-pin -q -c 0-383 ./src/qr_bench  546100000   12 2
likwid-pin -q -c 0-383 ./src/qr_bench  409600000   16 2
likwid-pin -q -c 0-383 ./src/qr_bench  204800000   32 2
likwid-pin -q -c 0-383 ./src/qr_bench  153600000   48 2
likwid-pin -q -c 0-383 ./src/qr_bench  102400000   64 2
likwid-pin -q -c 0-383 ./src/qr_bench   51200000  128 2
likwid-pin -q -c 0-383 ./src/qr_bench   25600000  256 2
likwid-pin -q -c 0-383 ./src/qr_bench   12800000  512 2
likwid-pin -q -c 0-383 ./src/qr_bench    6400000 1024 2
