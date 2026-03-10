#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 600
#SBATCH -p amd
#SBATCH --threads-per-core=1
#SBATCH --mem=200G
#SBATCH --output="%x-%j.out"

# source load_modules_pitts_spack-25.2.sh
# module purge
# unset LD_LIBRARY_PATH
# module load spack-user
# module load miniforge3
# eval "$(conda shell.bash hook)"
# conda activate /home/zoel_ml/conda_pytorch_env

for ((i=1; i <= 50; i++)); do
  srun taskset -c 0-127 python ../tt_from_dense_torchtt.py 2 30 $i 1
done
