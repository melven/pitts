#!/bin/bash

#plain='amd_tt_axpby_plain.sh-6988.out'
#plain_large='amd_tt_axpby_plain_large.sh-6989.out'
#plain_notsqr='amd_tt_axpby_plain_notsqr.sh-6990.out'
#plain_large_notsqr='amd_tt_axpby_plain_large_notsqr.sh-6991.out'
#left='amd_tt_axpby_leftnormalized.sh-6962.out'
#left_large='amd_tt_axpby_leftnormalized_large.sh-6971.out'
#right='amd_tt_axpby_rightnormalized.sh-6972.out'
#right_large='amd_tt_axpby_rightnormalized_large.sh-6973.out'

plain='amd_tt_axpby_plain.sh-69110.out'
plain_large='amd_tt_axpby_plain_large.sh-69111.out'
plain_notsqr='amd_tt_axpby_plain_notsqr.sh-69113.out'
plain_notsqr_large='amd_tt_axpby_plain_large_notsqr.sh-69112.out'
left='amd_tt_axpby_leftnormalized.sh-69107.out'
left_large='amd_tt_axpby_leftnormalized_large.sh-69108.out'
right='amd_tt_axpby_rightnormalized.sh-69109.out'
right_large='amd_tt_axpby_rightnormalized_large.sh-69098.out'


i=0
for f in plain_notsqr plain left right; do

  rm -f t1 t2 t3 t4

  f1=${!f}
  f_large=${f}_large
  f2=${!f_large}

  echo "# ${i}. $f"
  echo "# $f1 and $f2"

  echo "#   n   d   r1  r2    \"axpby runtime\"   \"axpby calls\"   \"left/rightNormalize runtime\"   \"left/rightNormalize calls\" \"normalize_qb runtime\" \"normalize_qb calls\" \"axpby_contract1 runtime\" \"axpby_contract1 calls"

  fgrep 'axpby(' $f1 | sed 's/.*] *//' | awk '{print "50 \t10 \t 50\t" NR "  \t" $1 "   \t" $2}' >> t1
  echo "" >> t1
  fgrep 'axpby(' $f2 | sed 's/.*] *//' | awk '{print "50 \t10 \t 50\t" NR+99 "  \t" $1 "   \t" $2}' >> t1

  fgrep 'Normalize(' $f1 | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t2
  echo "" >> t2
  fgrep 'Normalize(' $f2 | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t2

  fgrep 'normalize_qb(' $f1 | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t3
  echo "" >> t3
  fgrep 'normalize_qb(' $f2 | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t3

  fgrep 'axpby_contract1(' $f1 | grep -v "r1sum" | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t4
  echo "" >> t4
  fgrep 'axpby_contract1(' $f2 | grep -v "r1sum" | sed 's/.*] *//' | awk '{print "  \t" $1 "   \t" $2}' >> t4

  paste t1 t2 t3 t4
  echo ""
  echo ""


  ((i++))
done

