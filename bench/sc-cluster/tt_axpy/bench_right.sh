#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 300
#SBATCH --mem=10G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# 20^6, 20^8, 20^10, 50^8, 100^8
# Raenge 1-150

likwid-pin -c 0 ../../../build/src/axpby_right_ortho_bench  6  20 150  99  25 > bench_6_20_150_99_right.txt

likwid-pin -c 0 ../../../build/src/axpby_right_ortho_bench  8  20  70  50 120 > bench_8_20_70_50_right.txt

likwid-pin -c 0 ../../../build/src/axpby_right_ortho_bench 10  20  80  30 150 > bench_10_20_80_30_right.txt

likwid-pin -c 0 ../../../build/src/axpby_right_ortho_bench  8  50 100  10 120 > bench_8_50_100_10_right.txt

likwid-pin -c 0 ../../../build/src/axpby_right_ortho_bench  8 100  30  30  70 > bench_8_100_30_30_right.txt


likwid-pin -c 0-15 ../../../build/src/axpby_right_ortho_bench  6  20 150  99  25 > bench_6_20_150_99_right_omp16.txt

likwid-pin -c 0-15 ../../../build/src/axpby_right_ortho_bench  8  20  70  50 120 > bench_8_20_70_50_right_omp16.txt

likwid-pin -c 0-15 ../../../build/src/axpby_right_ortho_bench 10  20  80  30 150 > bench_10_20_80_30_right_omp16.txt

likwid-pin -c 0-15 ../../../build/src/axpby_right_ortho_bench  8  50 100  10 120 > bench_8_50_100_10_right_omp16.txt

likwid-pin -c 0-15 ../../../build/src/axpby_right_ortho_bench  8 100  30  30  70 > bench_8_100_30_30_right_omp16.txt
