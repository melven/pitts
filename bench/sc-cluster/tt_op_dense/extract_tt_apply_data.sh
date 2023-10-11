#!/bin/bash

eigensmall='tt_apply_op_dense__.sh-4006.out'
eigenlarge='tt_apply_op_dense_large__.sh-4008.out'
puremklsmall='tt_apply_op_dense.sh-4005.out'
puremkllarge='tt_apply_op_dense_large.sh-4138.out'
numpygemmsmall='tt_numpy_gemm.sh-4009.out'
numpygemmlarge='tt_numpy_gemm_large.sh-4010.out'

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

echo " 4. pure numpy GEMM of the same size ($numpygemmsmall and $numpagemmlarge)"
fgrep "gemm_numpy wtime" $numpygemmsmall | sed -n 'n;p' | sed 's/gemm_numpy wtime://' | paste -d " " - - - - - - | awk '{print NR "  " $1 "  " $2 "  "  $3 "  " $4 "  " $5 "  " $6 "  " 1000}'
#echo ""
fgrep "gemm_numpy wtime" $numpygemmlarge | sed -n 'n;p' | sed 's/gemm_numpy wtime://' | paste -d " " - - - - - - | awk '{print NR+99 "  " $1 "  " $2 "  "  $3 "  " $4 "  " $5 "  " $6 "  " 100}'
