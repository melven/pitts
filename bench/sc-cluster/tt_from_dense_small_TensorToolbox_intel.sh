#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 600
#SBATCH --mem=150G
#SBATCH --nodelist=be-cpu05
#SBATCH --output="%x-%j.out"

# module unload python
# export PATH=/scratch/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-10.2.0/miniconda2-4.7.12.1-tpknunsk2wg7pinqtras5szuc4ryrdqu/bin/:$PATH
# eval "$(conda shell.bash hook)"
# conda activate melven_TensorToolbox

for ((i=1; i <= 50; i++)); do
  python ../tt_from_dense_TensorToolbox.py 2 27 $i 1
done
