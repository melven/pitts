#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 300
#SBATCH --exclusive
#SBATCH --mem=10G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

likwid-pin -c 0 ../../../build/src/qr_bench > qr_bench_serial.txt

likwid-pin -c 0-15 ../../../build/src/qr_bench > qr_bench_omp16.txt