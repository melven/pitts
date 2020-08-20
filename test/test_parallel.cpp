#include <gtest/gtest.h>
#include "pitts_parallel.hpp"

namespace
{
  // helper function to determine the currently default number of threads in a parallel region
  int get_default_num_threads()
  {
    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp critical (PITTS_TEST_PARALLEL)
      numThreads = omp_get_num_threads();
    }
    return numThreads;
  }
}

TEST(PITTS_Parallel, ompThreadInfo_serial)
{
  ASSERT_LE(4, omp_get_max_threads());

  const auto& [iThread,nThreads] = PITTS::internal::parallel::ompThreadInfo();

  EXPECT_EQ(0, iThread);
  EXPECT_EQ(1, nThreads);
}

TEST(PITTS_Parallel, ompThreadInfo)
{
  ASSERT_LE(4, omp_get_max_threads());

  const auto nThreadsDefault = get_default_num_threads();
  ASSERT_LE(4, nThreadsDefault);

  std::vector<int> iThreads(nThreadsDefault);
  std::vector<int> nThreads(nThreadsDefault);

#pragma omp parallel for schedule(static)
  for(int i = 0; i < nThreadsDefault; i++)
  {
    const auto& [iT,nT] = PITTS::internal::parallel::ompThreadInfo();
    iThreads[i] = iT;
    nThreads[i] = nT;
  }

  std::vector<int> iThreads_ref(nThreadsDefault);
  std::vector<int> nThreads_ref(nThreadsDefault);
  for(int i = 0; i < nThreadsDefault; i++)
  {
    iThreads_ref[i] = i;
    nThreads_ref[i] = nThreadsDefault;
  }

  EXPECT_EQ(iThreads_ref, iThreads);
  EXPECT_EQ(nThreads_ref, nThreads);
}


TEST(PITTS_Parallel, mpiProcInfo_self)
{
  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo(MPI_COMM_SELF);
  EXPECT_EQ(0, iProc);
  EXPECT_EQ(1, nProcs);
}

TEST(PITTS_Parallel, mpiProcInfo)
{
  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();

  int iProc_ref = 0, nProcs_ref = 1;
  ASSERT_EQ(MPI_SUCCESS, MPI_Comm_rank(MPI_COMM_WORLD, &iProc_ref));
  ASSERT_EQ(MPI_SUCCESS, MPI_Comm_size(MPI_COMM_WORLD, &nProcs_ref));

  EXPECT_EQ(iProc_ref, iProc);
  EXPECT_EQ(nProcs_ref, nProcs);
}

// depending on the MPI vendor, this just kills the process with an error message - so disable it per default...
TEST(DISABLED_PITTS_Parallel, mpiProcInfo_error)
{
  ASSERT_THROW(PITTS::internal::parallel::mpiProcInfo(MPI_COMM_NULL), std::runtime_error);
}


TEST(PITTS_Parallel, distribute_zeroElems)
{
  for(int i = 0; i < 15; i++)
  {
    const auto& [firstElem,lastElem] = PITTS::internal::parallel::distribute(0, {i,15});
    EXPECT_EQ(0, firstElem);
    EXPECT_EQ(-1, lastElem);
  }
}

TEST(PITTS_Parallel, distribute_trivial)
{
  for(int i = 0; i < 5; i++)
  {
    const auto& [firstElem,lastElem] = PITTS::internal::parallel::distribute(15, {i,5});
    EXPECT_EQ(i*3, firstElem);
    EXPECT_EQ((i+1)*3-1, lastElem);
  }
}

TEST(PITTS_Parallel, distribute_withRemainder)
{
  const std::array<long long,5> nLocal = {7,7,6,6,6};
  const std::array<long long,5> offsets = {0,7,14,20,26};
  const long long nTotal = 7+7+6+6+6;

  for(int i = 0; i < 5; i++)
  {
    const auto& [firstElem,lastElem] = PITTS::internal::parallel::distribute(nTotal, {i,5});
    EXPECT_EQ(offsets[i], firstElem);
    EXPECT_EQ(offsets[i]+nLocal[i]-1, lastElem);
  }
}

TEST(PITTS_Parallel, distribute_tooManyProcs)
{
  for(int i = 0; i < 20; i++)
  {
    const auto& [firstElem,lastElem] = PITTS::internal::parallel::distribute(7, {i,20});
    if( i < 7 )
    {
      EXPECT_EQ(i, firstElem);
      EXPECT_EQ(i, lastElem);
    }
    else
    {
      EXPECT_EQ(7, firstElem);
      EXPECT_EQ(6, lastElem);
    }
  }
}


TEST(PITTS_Parallel, mpiGather)
{
  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
  const std::string localData = "Hello from process " + std::to_string(iProc) + ".\n";

  const int root = nProcs - 1;

  const auto& [globalData,offsets] = PITTS::internal::parallel::mpiGather(localData, root);

  std::string globalData_ref(0, '\0');
  std::vector<int> offsets_ref(nProcs+1, 0);

  if( iProc == root )
  {
    for(int i = 0; i < nProcs; i++)
    {
      globalData_ref += "Hello from process " + std::to_string(i) + ".\n";
      offsets_ref[i+1] = globalData_ref.size();
    }
  }

  EXPECT_EQ(globalData_ref, globalData);
  EXPECT_EQ(offsets_ref, offsets);
}


TEST(PITTS_Parallel, mpiCombineMaps)
{
  using StringMap = std::unordered_map<std::string,std::string>;

  StringMap localMap;
  localMap["hello"] = "world";

  const auto op = [](const std::string& s1, const std::string& s2){return s1 + " | " + s2;};

  int nProcs = 1, iProc = 0;
  ASSERT_EQ(0, MPI_Comm_size(MPI_COMM_WORLD, &nProcs));
  ASSERT_EQ(0, MPI_Comm_rank(MPI_COMM_WORLD, &iProc));

  StringMap globalMap = PITTS::internal::parallel::mpiCombineMaps(localMap, op);

  if( iProc == 0 )
  {
    ASSERT_EQ(1, globalMap.size());
    std::string str_ref = "world";
    for(int i = 1; i < nProcs; i++)
      str_ref = str_ref + " | world";
    ASSERT_EQ(str_ref, globalMap["hello"]);
  }
  else
  {
    ASSERT_EQ(0, globalMap.size());
    ASSERT_EQ(0, 0);
  }

  localMap.clear();
  localMap["proc"] = std::to_string(iProc);
  localMap["only_local: "+std::to_string(iProc)] = "I'm here";

  globalMap = PITTS::internal::parallel::mpiCombineMaps(localMap);

  if( iProc == 0 )
  {
    ASSERT_EQ(1+nProcs, globalMap.size());
    std::string str_ref = "";
    for(int i = 0; i < nProcs; i++)
      str_ref = str_ref + std::to_string(i);
    ASSERT_EQ(str_ref, globalMap["proc"]);
    for(int i = 0; i < nProcs; i++)
    {
      std::string key = "only_local: " + std::to_string(i);
      ASSERT_EQ("I'm here", globalMap[key]);
    }
  }
  else
  {
    // dummy checks as all checks are global
    ASSERT_EQ(0, globalMap.size());
    ASSERT_EQ(0, 0);
    for(int i = 0; i < nProcs; i++)
    {
      ASSERT_EQ(0, 0);
    }
  }
}
