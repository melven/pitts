#!/bin/bash

for f in *N1_n1*.out *N1_n2*.out *N1_n4*.out *N2_n8*.out *N4_n16*.out; do
  fgrep fromDense $f /dev/null | sed 's/tt_from_dense_thickbounds_d/ /' | sed 's/_N/  /' | sed 's/_n/  /' | sed 's/.sh-.*fromDense<double>/ /'
  echo ""
done
