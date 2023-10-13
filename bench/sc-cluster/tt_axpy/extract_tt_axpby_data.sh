#!/bin/bash

plain='amd_tt_axpby_plain.sh-6988.out'
plain_large='amd_tt_axpby_plain_large.sh-6989.out'
plain_notsqr='amd_tt_axpby_plain_notsqr.sh-6990.out'
plain_large_notsqr='amd_tt_axpby_plain_large_notsqr.sh-6991.out'
left='amd_tt_axpby_leftnormalized.sh-6962.out'
left_large='amd_tt_axpby_leftnormalized_large.sh-6971.out'
right='amd_tt_axpby_rightnormalized.sh-6972.out'
right_large='amd_tt_axpby_rightnormalized_large.sh-6973.out'



echo "# 0. plain (50^10, r1=50, r2=1...700, $plain_notsqr and $plain_large_notsqr)"
fgrep 'axpby(' $plain_notsqr | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'axpby(' $plain_large_notsqr | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 1. plain with TSQR (50^10, r1=50, r2=1...700, $plain and $plain_large)"
fgrep 'axpby(' $plain | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'axpby(' $plain_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 2. left-normalized with TSQR (50^10, r1=50, r2=1...700, $left and $left_large)"
fgrep 'axpby(' $left | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'axpby(' $left_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 3. right-normalized with TSQR (50^10, r1=50, r2=1...700, $right and $right_large)"
fgrep 'axpby(' $right | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'axpby(' $right_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""



echo "# 4. leftNormalize/SVD-Sweep for plain (50^10, r1=50, r2=1...700, $plain_notsqr and $plain_large_notsqr)"
fgrep 'leftNormalize(' $plain_notsqr | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'leftNormalize(' $plain_large_notsqr | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 5. leftNormalize/SVD-Sweep for plain with TSQR (50^10, r1=50, r2=1...700, $plain and $plain_large)"
fgrep 'leftNormalize(' $plain | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'leftNormalize(' $plain_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 6. rightNormalize/SVD-Sweep for left-normalized with TSQR (50^10, r1=50, r2=1...700, $left and $left_large)"
fgrep 'rightNormalize(' $left | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'rightNormalize(' $left_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""


echo "# 7. leftNormalize/SVD-Sweep for right-normalized with TSQR (50^10, r1=50, r2=1...700, $right and $right_large)"
fgrep 'leftNormalize(' $right | sed 's/.*] *//' | awk '{print NR "  \t" $1 "   \t" $2}'
echo ""
fgrep 'leftNormalize(' $right_large | sed 's/.*] *//' | awk '{print NR+99 "  \t" $1 "   \t" $2}'

echo ""
echo ""

