#include <gtest/gtest.h>
#include <omp.h>
#include "pitts_multivector_random.hpp"
#include "pitts_random.hpp"
#include <set>

namespace
{
  // helper function to determine the currently default number of threads in a parallel region
  int get_default_num_threads()
  {
    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp critical (PITTS_TEST_MULTIVECTOR_TSQR)
      numThreads = omp_get_num_threads();
    }
    return numThreads;
  }
}

TEST(PITTS_MultiVector_random, randomize)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double t2(3,20);

  randomize(t2);
  EXPECT_EQ(3, t2.rows());
  EXPECT_EQ(20, t2.cols());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(int i = 0; i < t2.rows(); i++)
    for(int j = 0; j < t2.cols(); j++)
        values.insert(t2(i,j));

  EXPECT_EQ(20*3, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(-1, v);
    EXPECT_LE(v, 1);
  }
}

TEST(PITTS_MultiVector_random, independent_of_number_of_threads)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using namespace PITTS;

  const int nThreads = get_default_num_threads();
  ASSERT_GE(nThreads, 4);

  MultiVector_double m1(20,3);
  MultiVector_double m2(20,3);
  MultiVector_double m3(20,3);
  MultiVector_double m4(20,3);

  const auto prevRngState = internal::randomGenerator;

  omp_set_num_threads(1);
  randomize(m1);
  const auto rngState1 = internal::randomGenerator;

  internal::randomGenerator = prevRngState;
  omp_set_num_threads(2);
  randomize(m2);
  const auto rngState2 = internal::randomGenerator;

  internal::randomGenerator = prevRngState;
  omp_set_num_threads(3);
  randomize(m3);
  const auto rngState3 = internal::randomGenerator;

  internal::randomGenerator = prevRngState;
  omp_set_num_threads(4);
  randomize(m4);
  const auto rngState4 = internal::randomGenerator;

  EXPECT_EQ(rngState1, rngState2);
  EXPECT_EQ(rngState2, rngState3);
  EXPECT_EQ(rngState3, rngState4);

  for(int i = 0; i < 20; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_EQ(m1(i,j), m2(i,j));
      EXPECT_EQ(m1(i,j), m3(i,j));
      EXPECT_EQ(m1(i,j), m4(i,j));
    }
}
