#include "pitts_common.hpp"
#include "pitts_parallel.hpp"
#include "pitts_multivector.hpp"
#include <stdio.h> // using printf because it behaves better when multithreading

using namespace PITTS;

void update_spaces(char* spaces, int depth, char space = ' ')
{
    if (depth >= 100 || depth < 0) {
        printf("\033[1;31mCRITICAL ERROR: depth value is out of range\033[0m");
        return;
    }
    for (int d = 0; d < depth; d++)
        spaces[d] = space;
    spaces[depth] = '\0';
}


void dummy_transformBlock(int m, const double* pdataIn, double* pdataResult)
{
    int nChunks = 1;
    int colBlockSize = 1;

    auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
    const int calc_color = 35;
    const int apply_color = 36;
    int depth = 0;
    char spaces[100];
    char antispaces[100];
    depth++;
    update_spaces(spaces, depth);

    // this is an approach for hierarchical blocking of
    // for(int i = 0; i < m; i++)                           // this order matters (need to do one i at a time, i+1 only after applied to it)
    //   calc i
    //   for(int j = i+1; j < m; j++)                       // this order doesn't matter (can apply i to any j anyhow)
    //     apply i to j

    const auto tree_apply = [&](const auto& tree_apply, int beginCol, int endCol, int applyBeginCol, int applyEndCol) -> void
    {
        depth++;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);

        int nCol = endCol - beginCol;
        // could also split by nApplyCol but doesn't seem to be help
        //int nApplyCol = applyEndCol - applyBeginCol;

        if( nCol < 2 )
        {
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tbase case ==> RETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol);
            //transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, beginCol, endCol, applyBeginCol, applyEndCol);
        }
        else
        {
            int middle = beginCol + nCol/2;
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse left (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, beginCol, middle, applyBeginCol, applyEndCol);
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse right (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, middle, endCol, applyBeginCol, applyEndCol);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tRETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol);

        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    const auto tree_calc = [&](const auto& tree_calc, int beginCol, int endCol) -> void
    {
        depth++;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);

        int nCol = endCol - beginCol;
        if( nCol < 2 )
        {
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tbase case ==> RETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol);
            for(int col = beginCol; col < endCol; col++)
            {
                //transformBlock_calc(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col);
                //transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col, col+1, col+1, endCol);
            }
        }
        else
        {
            int middle = beginCol + nCol/2;
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse left (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol, beginCol, middle);
            tree_calc(tree_calc, beginCol, middle);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tapplying columns %d-%d to %d-%d\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, middle, middle, endCol);
            tree_apply(tree_apply, beginCol, middle, middle, endCol);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse right (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol, middle, endCol);
            tree_calc(tree_calc, middle, endCol);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tRETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol);
        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    printf("%s\033[%dmtransform_block: \tthread %d: \tcombining columns %d to %d (all)\033[0m\n", spaces, calc_color, iThread, 0, m);
    tree_calc(tree_calc,0,m);
}


void dummy_block_TSQR(int nIter, int m)
{
    MultiVector<double> M(nIter, m);
    //Tensor2<double>& R;

    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_otherThreads(nMaxThreads);

    printf("--- BEGIN OF block_TSQR ---\n");

#pragma omp parallel
    {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
        // only work on a subset of threads if the input dimension is too small
        if( nIter < nThreads*2 )
            nThreads = 1 + (nIter-1)/2;
        
        std::unique_ptr<double[]> plocalBuff;
        if( iThread < nThreads )
        {
            plocalBuff.reset(new double[1]);
            if( nThreads > 1 )
                plocalBuff_otherThreads[iThread] = &plocalBuff[0];

            // fill with zero
            plocalBuff[0] = 0;

            const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

            printf("TSQR: \tloopdepth 0 \tthread %d: \tcombining blocks %lld to %lld (M)\n",iThread, firstIter, lastIter);
            for(long long iter = firstIter; iter <= lastIter; iter++)
            {
                dummy_transformBlock(m, &M(iter,0), &plocalBuff[0]);
            }
        }

        // tree reduction over threads
        int depth = 1;
        for(int nextThread = 1; nextThread < nThreads; nextThread*=2)
        {
#pragma omp barrier
#pragma omp master
            printf("\n* OMP BARRIER *\n\n");
#pragma omp barrier
            if( iThread % (nextThread*2) == 0 && iThread+nextThread < nThreads )
            {
                printf("TSQR: \tloopdepth %d \tthread %d: \tcombining block %d and %d (localBuffs)\n", depth, iThread, iThread, iThread+nextThread);
                const auto otherLocalBuff = plocalBuff_otherThreads[iThread+nextThread];
                dummy_transformBlock(m, &otherLocalBuff[0], &plocalBuff[0]);
            }
            //else
            //{
            //    printf("TSQR: \tloopdepth %d \tthread %d: \tdo nothing\n", depth, iThread);
            //}
        depth++;
        }

#pragma omp barrier
#pragma omp master
        printf("--- END OF block_TSQR ---\n");
    }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  int n = 32;
  int m = 8;

  dummy_block_TSQR(n, m);

  PITTS::finalize();

  return 0;
}