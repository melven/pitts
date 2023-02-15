#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 300
#SBATCH --exclusive
#SBATCH --mem=10G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

likwid-pin -c 0 ../../../build/src/qr_bench f > qr_bench_serial_fullrank.txt
likwid-pin -c 0 ../../../build/src/qr_bench h > qr_bench_serial_highrank.txt
likwid-pin -c 0 ../../../build/src/qr_bench l > qr_bench_serial_lowrank.txt

likwid-pin -c 0-15 ../../../build/src/qr_bench f > qr_bench_omp16_fullrank.txt
likwid-pin -c 0-15 ../../../build/src/qr_bench h > qr_bench_omp16_highrank.txt
likwid-pin -c 0-15 ../../../build/src/qr_bench l > qr_bench_omp16_lowrank.txt