#!/bin/bash

files="tt_from_dense_single_socket.log tt_from_dense_thickbounds_single_socket.log tt_from_dense_twosided_thickbounds_single_socket.log"
ranks="5 35 50"
for r in $ranks; do
  echo "# max_rank=$r"
  echo "# filename  total   calls  tsqr_total  tsqr1...10  tsmm_total  tsmm1...10  transpose_total  transpose1...10"
  for f in $files; do
    r2=$(expr 2 \* $r)
    total=$(fgrep "x $r2" -A 100 $f | grep fromDense | head -n 1 | awk '{print $3}')
    calls=$(fgrep "x $r2" -A 100 $f | grep fromDense | head -n 1 | awk '{print $4}')
    tsqr_total=$(fgrep "x $r2" -A 100 $f | grep block | grep -v rows | head -n 1 | awk '{print $3}')
    tsqr_steps=$(fgrep "x $r2" -A 100 $f | grep block | head -n 10 | awk '{print $8}')
    tsqr_steps=${tsqr_steps//$'\n'/ }
    tsmm_total=$(fgrep "x $r2" -A 100 $f | grep transform | grep -v Xrows | head -n 1 | awk '{print $3}')
    tsmm_steps=$(fgrep "x $r2" -A 100 $f | grep transform | head -n 10 | awk '{print $12}')
    tsmm_steps=${tsmm_steps//$'\n'/ }
    transpose_total=$(fgrep "x $r2" -A 100 $f | grep transpose | grep -v Xrows | head -n 1 | awk '{print $3}')
    transpose_steps=$(fgrep "x $r2" -A 100 $f | grep transpose | head -n 10 | awk '{print $12}')
    transpose_steps=${transpose_steps//$'\n'/ }
    echo "$f $total $calls $tsqr_total $tsqr_steps $tsmm_total $tsmm_steps $transpose_total $transpose_steps"
  done
done
