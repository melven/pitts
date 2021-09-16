#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 600
#SBATCH --exclusive
#SBATCH --output="%x-%j.out"

# module load miniconda2
# conda init bash
# conda activate resatensor2

for ((i=1; i <= 50; i++)); do
  srun python ../tt_from_dense_ttpy.py 2 27 $i 1
done
