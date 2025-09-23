#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

export LD_LIBRARY_PATH=/scratch/spack-25.2/opt/spack/linux-x86_64_v3/suite-sparse-7.7.0-7xi4fj5r5ftmrygrkmglh6ogfpc7v6fg/lib:/scratch/spack-25.2/opt/spack/linux-x86_64_v3/netlib-lapack-3.12.1-xwgsgfw6arpj4jz4cjfsm3rczf4e4r4c/lib:/scratch/spack-25.2/opt/spack/linux-x86_64_v3/boost-1.88.0-cjr6xk5fxe6cju2i3rrseoyonopyrfk7/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~/xerus:$PYTHONPATH

for ((i=1; i <= 50; i++)); do
  python ../tt_from_dense_xerus.py 2 27 $i 1
done
