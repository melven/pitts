#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_parallel.hpp"
#include "pitts_multivector.hpp"
#include <cstdio> // using printf because it behaves better when multithreading
#include <latch>

using namespace PITTS;

class Latch : public std::latch
{
    public:
        Latch(std::ptrdiff_t expected = 0) : std::latch(expected) {}
};

using LatchArray3 = std::array<Latch, 3>;

inline void resetLatch(Latch& l, std::ptrdiff_t expected)
{
    l.~Latch();
    new(&l) Latch(expected);
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


void update_spaces(char* spaces, int depth, char space = ' ')
{
    if (depth >= 100 || depth < 0) {
        //printf("\033[1;31mCRITICAL ERROR: depth value is out of range\033[0m");
        return;
    }
    for (int d = 0; d < depth; d++)
        spaces[d] = space;
    spaces[depth] = '\0';
}

void output_entries(double** M, int length, int m)
{
    //printf("\n");
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < m; j++)
        {
            //printf("%f\t", M[i][j]);
        }
        //printf("\n");
    }
    //printf("\n");
}

void check(const MultiVector<double>& M, int nThreads, int nIter, int m)
{
    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_allThreads(nMaxThreads);

    // only work on a subset of threads if the input dimension is too small
    if( nIter < nThreads*2 )
        nThreads = 1 + (nIter-1)/2;
        
    for (int iThread = 0; iThread < nThreads; iThread++)
    {
        double* plocalBuff = new double[m];
        // fill with zero
        for (int i = 0; i < m; i++)
            plocalBuff[i] = 0;
        plocalBuff_allThreads[iThread] = &plocalBuff[0];
    }

    for (int iThread = 0; iThread < nThreads; iThread++)
    {
        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});
        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
            const auto plocalBuff = plocalBuff_allThreads[iThread];
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
                const auto plocalBuff = plocalBuff_allThreads[iThread];
                const auto otherLocalBuff = plocalBuff_allThreads[iThread+nextThread];
                
                //dummy_transformBlock(m, &otherLocalBuff[0], &plocalBuff[0]);
                for(int i = 0; i < m; i++)
                {
                    dummy_transformBlock_calc(m, &otherLocalBuff[0], &plocalBuff[0], i);
                    dummy_transformBlock_apply(m, &otherLocalBuff[0], &plocalBuff[0], i, i+1, i+1, m);
                }
            }
        }
    }
    
    //printf("\nCheck entries:\n");
    output_entries(plocalBuff_allThreads.data(), nThreads, m);

    for (int iThread = 0; iThread < nThreads; iThread++)
        delete[] plocalBuff_allThreads[iThread];
}


void dummy_transformBlock(int m, const double* pdataIn, double* pdataResult)
{
    int nChunks = 1;
    int colBlockSize = 1;

    auto iThread = omp_get_thread_num();
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
            //printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tbase case ==> RETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol-1, applyBeginCol, applyEndCol-1);
            dummy_transformBlock_apply(m, pdataIn, pdataResult, beginCol, endCol, applyBeginCol, applyEndCol);
        }
        else
        {
            int middle = beginCol + nCol/2;
            //printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse left (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, beginCol, middle-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            //printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \trecurse right (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol, applyBeginCol, applyEndCol, middle, endCol-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
            //printf("%s\033[%dmtree_apply:%s\tthread %d: \tcolumns %d-%d -> %d-%d: \tRETURN\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, endCol-1, applyBeginCol, applyEndCol-1);

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
            //printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tbase case ==> RETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1);
            for(int col = beginCol; col < endCol; col++)
            {
                dummy_transformBlock_calc(m, pdataIn, pdataResult, col);
                dummy_transformBlock_apply(m, pdataIn, pdataResult, col, col+1, col+1, endCol);
            }
        }
        else
        {
            int middle = beginCol + nCol/2;
            //printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse left (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1, beginCol, middle-1);
            tree_calc(tree_calc, beginCol, middle);
            //printf("%s\033[%dmtree_calc:%s\tthread %d: \tapplying columns %d-%d to %d-%d\033[0m\n", spaces, apply_color, antispaces, iThread, beginCol, middle-1, middle, endCol-1);
            tree_apply(tree_apply, beginCol, middle, middle, endCol);
            //printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \trecurse right (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1, middle, endCol-1);
            tree_calc(tree_calc, middle, endCol);
            //printf("%s\033[%dmtree_calc:%s\tthread %d: \tcolumns %d to %d: \tRETURN\033[0m\n", spaces, calc_color, antispaces, iThread, beginCol, endCol-1);
        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    //printf("%s\033[%dmtransform_block: \tthread %d: \tcombining columns %d to %d (all)\033[0m\n", spaces, calc_color, iThread, 0, m-1);
    tree_calc(tree_calc,0,m);
}


/**
 * Exactly the threads [firstThread,lastThread) are expected to call this function with the same arguments.
 * The tree_apply recursion is parallelized orthogonally to the recursion (as existant before), in the sense that
 * it splits up [applyBeginCol,applyEndCol) onto the thread team firstThread,...,lastThread-1 the same way omp 
 * parallel for would (while the recursion is breaking up [beginCol,endCol]).
 * Every tree_apply is parallelized onto all threads in the team (no matter whether [applyBeginCol,applyEndCol)
 * is small or big).
 * Besides that parallelization, the implementation remains the same as before.
 */
void par_dummy_transformBlock(int m, const double* pdataIn, double* pdataResult, int firstThread, int lastThread, LatchArray3& bossLatches, LatchArray3& workerLatches)
{
    int nChunks = 1;
    int colBlockSize = 1;

    const int calc_color = 35;
    const int apply_color = 36;
    int depth = 0;
    char spaces[100];
    char antispaces[100];
    depth++;
    update_spaces(spaces, depth);

    // this is an approach for hierarchical blocking of
    // for(int i = 0; i < m; i++)                           // this order matters (need to do one i at a time, i+1 only after applied to it)
    //   calc i                                             // calc after being applied to and before applying to others
    //   for(int j = i+1; j < m; j++)                       // this order doesn't matter (can apply i to any j anyhow)
    //     apply i to j

    auto iThread = omp_get_thread_num();

    const auto tree_apply = [&](const auto& tree_apply, int beginCol, int endCol, int applyBeginCol, int applyEndCol) -> void
    {
        depth++;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);

        int nCol = endCol - beginCol;
        int nApplyCol = applyEndCol - applyBeginCol;

        if( nCol < 2 )
        {
            const int relThreadId = iThread - firstThread;
            const int nThreads = lastThread - firstThread;
            auto [localApplyBeginCol, localApplyEndCol] = internal::parallel::distribute(nApplyCol, {relThreadId, nThreads});
            localApplyBeginCol += applyBeginCol;
            localApplyEndCol += applyBeginCol + 1;
            
            //printf("%s\033[%dmtree_apply:%s\tthread %d (in team [%d,%d]): \tcolumns %d-%d -> %lld-%lld: \tparallel base case ==> RETURN\033[0m\n", spaces, apply_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, localApplyBeginCol, localApplyEndCol-1);
            dummy_transformBlock_apply(m, pdataIn, pdataResult, beginCol, endCol, localApplyBeginCol, localApplyEndCol);
        }
        else
        {
            int middle = beginCol + nCol/2;
            //printf("%s\033[%dmtree_apply:%s\tthread %d (in team [%d,%d]): \tcolumns %d-%d -> %d-%d: \trecurse left (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, applyBeginCol, applyEndCol-1, beginCol, middle-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            //printf("%s\033[%dmtree_apply:%s\tthread %d (in team [%d,%d]): \tcolumns %d-%d -> %d-%d: \trecurse right (col %d-%d -> %d-%d)\033[0m\n", spaces, apply_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, applyBeginCol, applyEndCol-1, middle, endCol-1, applyBeginCol, applyEndCol-1);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
            //printf("%s\033[%dmtree_apply:%s\tthread %d (in team [%d,%d]): \tcolumns %d-%d -> %d-%d: \tRETURN\033[0m\n", spaces, apply_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, applyBeginCol, applyEndCol-1);
        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    int globalCounter = 0;

    const auto tree_calc = [&](const auto& tree_calc, int beginCol, int endCol) -> void
    {
        depth++;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);

        int nCol = endCol - beginCol;
        if( nCol < 2 )
        {
            if (iThread == firstThread)
            {
                //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tcolumns %d to %d: \tbase case (do calc) ==> RETURN\033[0m\n", spaces, calc_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1);
                for(int col = beginCol; col < endCol; col++)
                {
                    dummy_transformBlock_calc(m, pdataIn, pdataResult, col);
                    dummy_transformBlock_apply(m, pdataIn, pdataResult, col, col+1, col+1, endCol);
                }
            }
            else
            {
                //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tcolumns %d to %d: \tbase case (do nothing) ==> RETURN\033[0m\n", spaces, calc_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1);
            }
        }
        else
        {
            int middle = beginCol + nCol/2;

            //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tcolumns %d to %d: \trecurse left (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, beginCol, middle-1);
            tree_calc(tree_calc, beginCol, middle);

            const int cnt = globalCounter++;
            if( iThread == firstThread )
                bossLatches[cnt%3].count_down();
            else
                bossLatches[cnt%3].wait();

            //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tapplying columns %d-%d to %d-%d\033[0m\n", spaces, apply_color, antispaces, iThread, firstThread, lastThread-1, beginCol, middle-1, middle, endCol-1);
            tree_apply(tree_apply, beginCol, middle, middle, endCol);

            if( iThread != firstThread )
                workerLatches[cnt%3].count_down();
            
            if( iThread == firstThread )
            {
                workerLatches[cnt%3].wait();
                resetLatch(bossLatches[(cnt+2)%3], 1);
                const int nThreads = lastThread - firstThread;
                resetLatch(workerLatches[(cnt+2)%3], nThreads-1);
            }
            //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tcolumns %d to %d: \trecurse right (col %d to %d)\033[0m\n", spaces, calc_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1, middle, endCol-1);
            tree_calc(tree_calc, middle, endCol);

            //printf("%s\033[%dmtree_calc:%s\tthread %d (in team [%d,%d]): \tcolumns %d to %d: \tRETURN\033[0m\n", spaces, calc_color, antispaces, iThread, firstThread, lastThread-1, beginCol, endCol-1);
        }

        depth--;
        update_spaces(spaces, depth);
        update_spaces(antispaces, 10 - depth);
    };

    //printf("%s\033[%dmtransform_block: \tthread %d (in team [%d,%d]): \tcombining columns %d to %d (all)\033[0m\n", spaces, calc_color, iThread, firstThread, lastThread-1, 0, m-1);
    tree_calc(tree_calc,0,m);
}


int dummy_block_TSQR(const MultiVector<double>& M, int nIter, int m)
{
    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_allThreads(nMaxThreads);

    //printf("--- BEGIN OF block_TSQR ---\n");

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
                plocalBuff_allThreads[iThread] = &plocalBuff[0];

            // fill with zero
            for (int i = 0; i < m; i++)
                plocalBuff[i] = 0;

            const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

            //printf("TSQR: \tloopdepth 0 \tthread %d: \tcombining blocks %lld to %lld (M)\n",iThread, firstIter, lastIter);
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
        {
            //printf("\n* OMP BARRIER *\n\n");
        }
#pragma omp barrier
            if( iThread % (nextThread*2) == 0 && iThread+nextThread < nThreads )
            {
                //printf("TSQR: \tloopdepth %d \tthread %d: \tcombining block %d and %d (localBuffs)\n", depth, iThread, iThread, iThread+nextThread);
                const auto otherLocalBuff = plocalBuff_allThreads[iThread+nextThread];
                dummy_transformBlock(m, &otherLocalBuff[0], &plocalBuff[0]);
            }
            //else
            //{
            //    //printf("TSQR: \tloopdepth %d \tthread %d: \tdo nothing\n", depth, iThread);
            //}
        depth++;
        }

#pragma omp barrier
#pragma omp master
        {
            //printf("--- END OF block_TSQR ---\n");
            output_entries(plocalBuff_allThreads.data(), numthreads, m);
        }
#pragma omp barrier // needed in order for other plocalBuff's memory (and hence memory pointed to by plocalBuff_allThreads) not to be destroyed prematurely
    }
    return numthreads;
}

/** manuel thread teams parallelization:
 * The (participating!) threads are split into teams, with each team working on a specific block.
 * All (participating!) threads call transformBlock (par_dummy_transformBlock), and each member of a
 * team calls it with the exact same parameters (including a characterization of its team).
 * Teams are chosen naturally based on the preexisting code, i.e.
 * - for the first iteration: [0,1], [2,3], [3,4], ...
 * - second iteration:        [0,3], [4,7], ...
 * - ...
 * where care is taken for edge cases.
 * Inside transformBlock, the work can be split up onto the threads that call it.
 * This dummy function returns the number of participating threads, to be used for diagnostic output.
 */
int par_dummy_block_TSQR(const MultiVector<double>& M, int nIter, int m)
{
    int nMaxThreads = omp_get_max_threads();
    std::vector<double*> plocalBuff_allThreads(nMaxThreads);

    std::unique_ptr<LatchArray3[]> bossLatchBuff(new LatchArray3[nMaxThreads*2]);
    std::unique_ptr<LatchArray3[]> workerLatchBuff(new LatchArray3[nMaxThreads*2]);
    std::array<LatchArray3*,2> localBossLatches = {&bossLatchBuff[0], &bossLatchBuff[nMaxThreads]};
    std::array<LatchArray3*,2> localWorkerLatches = {&workerLatchBuff[0], &workerLatchBuff[nMaxThreads]};

    //printf("--- BEGIN OF block_TSQR ---\n");

    int numthreads; // variable only used to be able to return nThreads
#pragma omp parallel
    {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
        // only work on a subset of threads if the input dimension is too small (specifically: we cap nThreads to ⌈nIter/2⌉)
        if( nIter < nThreads*2 )
            nThreads = 1 + (nIter-1)/2;
#pragma omp master
        numthreads = nThreads;
        
        std::unique_ptr<double[]> plocalBuff;
        if (iThread < nThreads)
        {
            plocalBuff.reset(new double[m]);
            // if (nThreads > 1) this if statement not really needed: no gain from checking this just to save one assignment in special case
            plocalBuff_allThreads[iThread] = &plocalBuff[0];

            // fill with zero
            for (int i = 0; i < m; i++)
                plocalBuff[i] = 0;

            const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

            // no synchronization after creation needed here because each thread uses exactly its own barrier
            resetLatch(localBossLatches[0][iThread][0], 1);
            resetLatch(localBossLatches[0][iThread][1], 1);
            resetLatch(localBossLatches[0][iThread][2], 1);
            resetLatch(localWorkerLatches[0][iThread][0], 0);
            resetLatch(localWorkerLatches[0][iThread][1], 0);
            resetLatch(localWorkerLatches[0][iThread][2], 0);

            //printf("TSQR: \tloopdepth 0 \tthreads [%d,%d]: \tcombining blocks %lld to %lld (M)\n", iThread, iThread, firstIter, lastIter);
            for(long long iter = firstIter; iter <= lastIter; iter++)
            {
                par_dummy_transformBlock(m, &M(iter,0), &plocalBuff[0], iThread, iThread+1, localBossLatches[0][iThread], localWorkerLatches[0][iThread]);
                 // barrier is not really needed here, but it simplifies code a little bit
                 // it can be left away if the overhead is significant (enough)
            }
            // above barriers are destructed at bottom of first loop iteration of the next loop (setting wasBossThread accordingly below)
        }

        // tree reduction over threads
        bool wasBossThread = (iThread < nThreads); // keep track of last iterations boss threads
        for(int nextThread = 1, cnt = 1; nextThread < nThreads; nextThread*=2, cnt++)
        {
            int bossThread, lastThread; // first (including) and last (excluding) threads in the team

            if (iThread < nThreads)
            {
                const int threadteamSize = nextThread*2; // actual thread team size is lastThread-bossThread
                bossThread = iThread - iThread % threadteamSize;
                lastThread = std::min(bossThread + threadteamSize, nThreads);
                // in following special case, could add following unused threads
                //if (lastThread < nThreads && bossThread+3*nextThread >= nThreads)
                //    lastThread = nThreads;

                // create local barrier before global barrier to ensure all threads in team have the same view of them
                if (iThread == bossThread) // iThread < nThreads s.t. lastThread-bossThread non-negative
                {
                    resetLatch(localBossLatches[cnt%2][bossThread][0], 1);
                    resetLatch(localBossLatches[cnt%2][bossThread][1], 1);
                    resetLatch(localBossLatches[cnt%2][bossThread][2], 1);
                    resetLatch(localWorkerLatches[cnt%2][bossThread][0], lastThread-bossThread - 1);
                    resetLatch(localWorkerLatches[cnt%2][bossThread][1], lastThread-bossThread - 1);
                    resetLatch(localWorkerLatches[cnt%2][bossThread][2], lastThread-bossThread - 1);
                }
            }
//#pragma omp barrier
//#pragma omp master
//        {
//            //printf("\n* OMP BARRIER *\n\n");
//        }
#pragma omp barrier
            if (iThread < nThreads)
            {
                if (bossThread+nextThread < nThreads)
                {
                    //printf("TSQR: \tloopdepth %d \tthreads [%d,%d] (I'm thread %d): \tcombining block %d and %d (localBuffs)\n", cnt, bossThread, lastThread-1, iThread, bossThread, bossThread+nextThread);
                    const auto bossLocalBuff = plocalBuff_allThreads[bossThread];
                    const auto otherLocalBuff = plocalBuff_allThreads[bossThread+nextThread];
                    par_dummy_transformBlock(m, &otherLocalBuff[0], &bossLocalBuff[0], bossThread, lastThread, localBossLatches[cnt%2][bossThread], localWorkerLatches[cnt%2][bossThread]);
                }

                // destruct last iterations local barriers (we wait for one iteration to ensure that there is a global barrier between last use and destruction of the barrier)
                // remark: explicit destruction only needed if the destructor has side effects
                if (wasBossThread)
                {
                    // remember this iterations boss threads (in order to correctly destruct barriers in next iteration)
                    // remark: here we are using the fact that this iteration's boss threads are a subset of last iteration's boss threads (hence, this minimal update suffices)
                    if (iThread != bossThread)
                        wasBossThread = false;
                }
            }
        }

//#pragma omp barrier
//#pragma omp master
//        {
//            //printf("--- END OF block_TSQR ---\n");
//            output_entries(plocalBuff_allThreads.data(), numthreads, m);
//        }
#pragma omp barrier // needed in order for other thread's plocalBuff (and hence memory pointed to by plocalBuff_allThreads) not to be destroyed prematurely
    }

    return numthreads;
}


int main(int argc, char* argv[])
{
    // parameters:
    // n: proportional to number of threads, nThreads = ⌈n/2⌉
    // m: number of rows
    constexpr bool test_extensive = true;
    constexpr bool benchmark = true;

    PITTS::initialize(&argc, &argv);

    if (!test_extensive && !benchmark)
    {
        const int n = 13;
        const int m = 8;
        MultiVector<double> M(n, m);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                M(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;

        const int numThreadsUsed = par_dummy_block_TSQR(M, n, m);

        check(M, numThreadsUsed, n, m);
    }

    if (test_extensive)
    {
        const int n_low = 1;
        const int n_upp = 32;
        const int m_low = 1;
        const int m_upp = 8;
        for (int m = m_low; m < m_upp; m++)
        {
            for (int n = n_low; n < n_upp; n++)
            {
                MultiVector<double> M(n, m);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        M(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;

                const int numThreadsUsed = par_dummy_block_TSQR(M, n, m);

                check(M, numThreadsUsed, n, m);
            }
        }
    }

    if (benchmark)
    {
        const int n_low = 2;
        const int n_upp = 1000;
        const int m_low = 1;
        const int m_upp = 100;
        printf("\nWalltime (in milliseconds) for different numbers of threads\n\n");
        for (int m = m_low; m < m_upp; m*=2)
            printf("\t\tm = %d", m);
        printf("\n\t\t_____________________________________________________________________________________________________");
        const int tot_it = 10;
        int numThreadsUsed;
        MultiVector<double> Mwarm(1000, 1000);
        for (int i = 0; i < 1000; i++)
            for (int j = 0; j < 1000; j++)
                Mwarm(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;
        // BENCH NEW VERSION
        printf("\n\nNEW PARALLEL VERSION:\n");
        // warmup
        numThreadsUsed = par_dummy_block_TSQR(Mwarm, 1000, 1000);
        // measurement loop
        for (int n = n_low; n < n_upp; n*=2)
        {
            const int _nThreads = std::min(1 + (n-1)/2, omp_get_max_threads());
            printf("\n#threads=%d ", _nThreads);
            if (_nThreads < 10) printf(" ");
            printf("|\t");

            for (int m = m_low; m < m_upp; m*=2)
            {
                MultiVector<double> M(n, m);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        M(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;

                double wtime = omp_get_wtime();
                for (int i = 0; i < tot_it; i++)
                    numThreadsUsed = par_dummy_block_TSQR(M, n, m);
                wtime = omp_get_wtime() - wtime;

                printf("%f\t", wtime*1000);
            }
        }
        // BENCH OLD VERSION
        printf("\n\nOLD \"SERIAL\" VERSION:\n");
        // warmup
        numThreadsUsed = dummy_block_TSQR(Mwarm, 1000, 1000);
        // measurement loop
        for (int n = n_low; n < n_upp; n*=2)
        {
            const int _nThreads = std::min(1 + (n-1)/2, omp_get_max_threads());
            printf("\n#threads=%d ", _nThreads);
            if (_nThreads < 10) printf(" ");
            printf("|\t");

            for (int m = m_low; m < m_upp; m*=2)
            {
                MultiVector<double> M(n, m);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        M(i,j) = (((i+1)*7001+(j+1)*7919) % 101)/10;

                double wtime = omp_get_wtime();
                for (int i = 0; i < tot_it; i++)
                    numThreadsUsed = dummy_block_TSQR(M, n, m);
                wtime = omp_get_wtime() - wtime;

                printf("%f\t", wtime*1000);
            }
        }
        printf("\n\n");
    }

    PITTS::finalize();
    return 0;
}
