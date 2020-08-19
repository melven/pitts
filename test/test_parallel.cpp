#include <gtest/gtest.h>
#include "pitts_parallel.hpp"


TEST(PITTS_Parallel, combineMaps)
{
  using StringMap = std::unordered_map<std::string,std::string>;

  StringMap localMap;
  localMap["hello"] = "world";

  const auto op = [](const std::string& s1, const std::string& s2){return s1 + " | " + s2;};

  int nProcs = 1, iProc = 0;
  ASSERT_EQ(0, MPI_Comm_size(MPI_COMM_WORLD, &nProcs));
  ASSERT_EQ(0, MPI_Comm_rank(MPI_COMM_WORLD, &iProc));

  StringMap globalMap = PITTS::internal::parallel::combineMaps(localMap, op);

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

  globalMap = PITTS::internal::parallel::combineMaps(localMap);

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
