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

  fgrep 'Runtime (RDTSC)' $f | grep -v '|.*|.*|.*|.*|.*|.*|' | sed 's/STAT *|[ 0-9.]*|[ 0-9.]*|[ 0-9.]*//' | awk '{print $6}' >> t1
  fgrep "  DP [MFLOP" $f | grep -v '|.*|.*|.*|.*|.*|.*|' | sed 's/STAT//' | awk '{print $5}' >> t2
  fgrep unhalted $f | grep -v '|.*|.*|.*|.*|.*|.*|' | sed 's/STAT//' | awk '{print $6}' >> t3
  gmresIterText=$(fgrep 'apply(const TTOpApplyDenseHelper' $f)
  if [ "$?" != "0" ]; then
    gmresIterText=$(fgrep 'apply(const TensorTrainOperator<T>&, const MultiVector<T>&, MultiVector<T>&)' $f)
  fi
  if [ "$?" != "0" ]; then
    gmresIterText=$(fgrep 'apply(const TensorTrainOperator' $f)
  fi
  echo "$gmresIterText" | sed 's/.*double] *//' | awk '{print $2}' >> t4_

  fgrep 'GMRES(' $f | sed 's/.*double.*] *//' | awk '{print $2}' >> t5

  #n="$n $(cat t1 | wc -l)"
done

paste t4_ t5 | awk '{print $1-$2}' > t4
paste t1 t2 | awk '{print $1*$2}' > t6

# n0=$(echo "$n" | awk '{print $1}')
# n1=0
# if [ "$2" != "" ]; then
#   n1=$(echo "$n" | awk '{print $2-$1}')
# fi
# #echo "n: $n"
# #echo "n0: $n0, n1: $n1"
# 
# head -n $n0 t0 > t0_
# tail -n $n1 t0 >> t0_

echo '# n d r_rhs      runtime   MFLOP/s   runtime_unhalted  "total GMRES iter"  "GMRES calls" "Total MFlops"'
paste t0_ t1 t2 t3 t4 t5 t6
