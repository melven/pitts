#!/bin/bash

eigensmall="amd_no_direct_mkl_tt_apply_op_dense.sh-412915.out"
eigenlarge="amd_no_direct_mkl_tt_apply_op_dense_large.sh-412924.out"
puremklsmall="amd_tt_apply_op_dense.sh-412899.out"
puremkllarge="amd_tt_apply_op_dense_large.sh-412900.out"
numpygemmsmall="amd_tt_numpy_gemm.sh-412918.out"
numpygemmlarge="amd_tt_numpy_gemm_large.sh-412919.out"
torchsmall='amd_torch_tt_op_dense.sh-412925.out'
torchlarge='amd_torch_tt_op_dense_large.sh-412926.out'



echo "# 0. standard (Eigen -DEIGEN_USE_MKL, $eigensmall and $eigenlarge)"
fgrep 'apply(' $eigensmall | grep  -v TTOpApply | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
#echo ""
fgrep 'apply(' $eigenlarge | grep  -v TTOpApply | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""

echo "# 1. optimized (Eigen -DEIGEN_USE_MKL, $eigensmall and $eigenlarge)"
fgrep 'apply(' $eigensmall | grep  TTOpApply | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
#echo ""
fgrep 'apply(' $eigenlarge | grep  TTOpApply | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""

echo "# 2. standard (pure MKL, $puremklsmall and $puremkllarge)"
fgrep 'apply(' $puremklsmall | grep  -v TTOpApply | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
#echo ""
fgrep 'apply(' $puremkllarge | grep  -v TTOpApply | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""

echo "# 3. optimized (pure MKL, $puremklsmall and $puremkllarge)"
fgrep 'apply(' $puremklsmall | grep  TTOpApply | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
#echo ""
fgrep 'apply(' $puremkllarge | grep  TTOpApply | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""

echo " 4. pure numpy GEMM of the same size ($numpygemmsmall and $numpygemmlarge)"
fgrep "gemm_numpy wtime" $numpygemmsmall | sed -n 'n;p' | sed 's/gemm_numpy wtime://' | paste -d " " - - - - - - | awk '{print NR "  " $1 "  " $2 "  "  $3 "  " $4 "  " $5 "  " $6 "  " 1000}'
#echo ""
fgrep "gemm_numpy wtime" $numpygemmlarge | sed -n 'n;p' | sed 's/gemm_numpy wtime://' | paste -d " " - - - - - - | awk '{print NR+99 "  " $1 "  " $2 "  "  $3 "  " $4 "  " $5 "  " $6 "  " 100}'

echo ""
echo ""

echo "# 5. pytorch (2.13.0) torch.einsum (MKL 2026.1.0)"
echo "# files: $torchsmall and $torchlarge"
fgrep "tt_op_apply_torch wtime:" $torchsmall | sed 's/tt_op_apply_torch wtime://' | awk 'NR % 2 == 0' | awk '{print NR "  \t" $1 "   \t" 1000}'
fgrep "tt_op_apply_torch wtime:" $torchlarge | sed 's/tt_op_apply_torch wtime://' | awk 'NR % 2 == 0' | awk 'NR <= 200 {print NR+99 "  \t" $1 "   \t" 100}'
fgrep "tt_op_apply_torch wtime:" $torchlarge | sed 's/tt_op_apply_torch wtime://' | awk 'NR % 2 == 0' | awk 'NR > 200 && NR <= 400 {print NR+99 "  \t" $1 "   \t" 50}'
fgrep "tt_op_apply_torch wtime:" $torchlarge | sed 's/tt_op_apply_torch wtime://' | awk 'NR % 2 == 0' | awk 'NR > 400 {print NR+99 "  \t" $1 "   \t" 25}'


#for((r=100;r<300;r++)); do
#  echo "r: $r, nIter: 100"
#  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../torch_tt_op_dense_bench.py 50 $r 2 100
#done
#
#for((r=300;r<500;r++)); do
#  echo "r: $r, nIter: 50"
#  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../torch_tt_op_dense_bench.py 50 $r 2 50
#done
#
#for((r=500;r<=700;r++)); do
#  echo "r: $r, nIter: 25"
#  srun taskset -c 0-127 likwid-pin -c 0-63 python ../../torch_tt_op_dense_bench.py 50 $r 2 25
#done
