<!--
Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
SPDX-FileContributor: Manuel Joey Becklas

SPDX-License-Identifier: BSD-3-Clause
-->

NOTES ON THE BENCHMARK RESULTS

For some reason, striding the local barriers only has a negative effect on the dummy bench runtimes.
Why, is unclear to me.

-------------------------------------------------------------------------------------------------------------------------

The question is, why is the new version so much slower for larger m? (Even though larger m shouldn't affect synchronization overhead)

I doubt that there is any false sharing. The shared variables M, nIter, m are only read from. The only remaining shared variable plocalBuff_allThreads is initialized once and the memory it points to, ie the plocalBuff's, are allocated individually with new. (And the local barriers are allocated with a stride of 4KB.)

However, I think there is some true sharing within the calculation which might cause the slow down for increasing m. Recall the excel sheet with the arrows drawn etc: In one apply phase, each thread only writes to one block, but may read from many blocks, each of which have been written to previously. 
Therefore, quite a lot of data needs to be exchanged between the cores in each parallel apply phase, and the amount of this data grows with m. The amount of computation grows as well for increasing m, but (I assume) is much smaller than the sharing overhead to begin with.

This would explain the results rather well. In the old version, the walltime increases as the number of threads does (overhead from omp barrier synchronization increases) as well as with m after it becomes significant (amount of computation increases). In the new version, the same happens, but the increase with m is much more dramatic, as there is more sharing of data between the threads. For both versions, for m=1, the walltimes are the same for any amount of threads.

In the actual tsqr implementation these results may very well be different, because there, the computational intensity is much higher, and therefore the impact of transferring some data between the cores is much lower. But maybe, the block size will have to be adjusted to account for the larger synchronization overhead.