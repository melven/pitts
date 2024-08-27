#!/bin/bash

if [ "$1" == "" ]; then
  echo "Missing filename argument!"
  exit
fi

rm -f t0_ t1 t2 t3 t4_ t4 t5
#n=""

for f in $*; do
  echo "# $f"

  fgrep '# Arguments' $f | sed 's/.*(n=//' | sed 's/, d=/ /' | sed "s/, I=.*rhs_type='/ /" | sed "s/', rhs_random_rank=/ /" | sed 's/, lhs_type.*//' | sed 's/rhs_ones.*/ones/' | sed 's/rhs_random //' >> t0_

  fgrep 'solveMALS(' $f | sed 's/.*double. *\t*//' | awk '{print $1}' >> t1
  fgrep 'normalize_qb(' $f | sed 's/.*double. *\t*//' >> t2
  fgrep 'apply(' $f | grep MultiVector | sed 's/.*double. *\t*//' >> t3
  fgrep 'normalize_svd(' $f | sed 's/.*double. *\t*//' >> t4
done

echo '# n d r_rhs      runtime   "normalize_qb runtime"    "normalize_qb calls" "apply(TTOp, MultiVec) runtime"    "apply(TTOp, MultiVec) calls"    "normalize_svd runtime"    "normalize_svd calls"'
paste t0_ t1 t2 t3 t4
