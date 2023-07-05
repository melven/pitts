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

void output_entries(double** M, int length, int m)
{
    putchar('\n');
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%f\t", M[i][j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void check(MultiVector<double>& M, int nThreads, int nIter, int m)
{
    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_otherThreads(nMaxThreads);

    // only work on a subset of threads if the input dimension is too small
    if( nIter < nThreads*2 )
        nThreads = 1 + (nIter-1)/2;
        
    for (int iThread = 0; iThread < nThreads; iThread++)
    {
        double* plocalBuff = new double[m];
        // fill with zero
        for (int i = 0; i < m; i++)
            plocalBuff[i] = 0;
        plocalBuff_otherThreads[iThread] = &plocalBuff[0];
    }

    for (int iThread = 0; iThread < nThreads; iThread++)
    {
        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});
        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
            const auto plocalBuff = plocalBuff_otherThreads[iThread];
            //dummy_transformBlock(m, &M(iter,0), &plocalBuff[0]);
            for(int i = 0; i < m; i++)
            {
                dummy_transformBlock_calc(m, &M(iter,0), &plocalBuff[0], i);
                dummy_transformBlock_apply(m, &M(iter,0), &plocalBuff[0], i, i+1, i+1, m);
            }
        }
    }

    for(int nextThread = 1; nextThread < nThreads; nextThread*=2)
    {
        for (int iThread = 0; iThread < nThreads; iThread++)
        {
            if( iThread % (nextThread*2) == 0 && iThread+nextThread < nThreads )
            {
                const auto plocalBuff = plocalBuff_otherThreads[iThread];
                const auto otherLocalBuff = plocalBuff_otherThreads[iThread+nextThread];
                
                //dummy_transformBlock(m, &otherLocalBuff[0], &plocalBuff[0]);
                for(int i = 0; i < m; i++)
                {
                    dummy_transformBlock_calc(m, &otherLocalBuff[0], &plocalBuff[0], i);
                    dummy_transformBlock_apply(m, &otherLocalBuff[0], &plocalBuff[0], i, i+1, i+1, m);
                }
            }
        }
    }
    
    printf("\nCheck entries:\n");
    output_entries(plocalBuff_otherThreads.data(), nThreads, m);

    for (int iThread = 0; iThread < nThreads; iThread++)
        delete[] plocalBuff_otherThreads[iThread];
}


void dummy_transformBlock_calc(int m, const double* pdataIn, double* pdataResult, int col)
{
    pdataResult[col] = pdataIn[col] + pdataResult[col];
}

void dummy_transformBlock_apply(int m, const double* pdataIn, double* pdataResult, int beginCol, int endCol, int applyBeginCol, int applyEndCol)
{
    for (int i = beginCol; i < endCol; i++)
        for (int j = applyBeginCol; j < applyEndCol; j++)
            pdataResult[j] = 0.5*pdataResult[j] + 0.5*pdataIn[i];

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
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tbase case ==> RETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol-1, applyBeginCol, applyEndCol-1);
            dummy_transformBlock_apply(m, pdataIn, pdataResult, beginCol, endCol, applyBeginCol, applyEndCol);
        }
        else
        {
            int middle = beginCol + nCol/2;
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse left (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, beginCol, middle-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse right (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, middle, endCol-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
            printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tRETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol-1, applyBeginCol, applyEndCol-1);

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
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tbase case ==> RETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1);
            for(int col = beginCol; col < endCol; col++)
            {
                dummy_transformBlock_calc(m, pdataIn, pdataResult, col);
                dummy_transformBlock_apply(m, pdataIn, pdataResult, col, col+1, col+1, endCol);
            }
        }
        else
        {
            int middle = beginCol + nCol/2;
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse left (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1, beginCol, middle-1);
            tree_calc(tree_calc, beginCol, middle);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tapplying columns %d-%d to %d-%d\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, middle-1, middle, endCol-1);
            tree_apply(tree_apply, beginCol, middle, middle, endCol);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse right (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1, middle, endCol-1);
            tree_calc(tree_calc, middle, endCol);
            printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tRETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1);
        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    printf("%s\033[%dmtransform_block: \tthread %d: \tcombining columns %d to %d (all)\033[0m\n", spaces, calc_color, iThread, 0, m-1);
    tree_calc(tree_calc,0,m);
}


void dummy_block_TSQR(int nIter, int m)
{
    MultiVector<double> M(nIter, m);
    for (int i = 0; i < nIter; i++)
        for (int j = 0; j < m; j++)
            M(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;

    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_otherThreads(nMaxThreads);

    printf("--- BEGIN OF block_TSQR ---\n");

    int numthreads;
#pragma omp parallel
    {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
        // only work on a subset of threads if the input dimension is too small
        if( nIter < nThreads*2 )
            nThreads = 1 + (nIter-1)/2;
        numthreads = nThreads;
        
        std::unique_ptr<double[]> plocalBuff;
        if( iThread < nThreads )
        {
            plocalBuff.reset(new double[m]);
            if( nThreads > 1 )
                plocalBuff_otherThreads[iThread] = &plocalBuff[0];

            // fill with zero
            for (int i = 0; i < m; i++)
                plocalBuff[i] = 0;

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
        {
            printf("--- END OF block_TSQR ---\n");
            output_entries(plocalBuff_otherThreads.data(), numthreads, m);
        }
#pragma omp barrier // needed in order for other plocalBuff's memory (and hence memory pointed to by plocalBuff_otherThreads) not to be destroyed prematurely
    }
    check(M, numthreads, nIter, m);
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