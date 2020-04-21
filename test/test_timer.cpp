#include <gtest/gtest.h>
#include "pitts_timer.hpp"

namespace
{
  void simpleFunction(int x, int y)
  {
    PITTS::timing::ScopedTimer timer;
    printf("x+y: %d\n", x+y);
  }

  void simpleFunctionWithType(int x, int y)
  {
    PITTS::timing::ScopedTimer timer(x);
    printf("x+y: %d\n", x+y);
  }

}

TEST(PITTS_Timer, internal_TimingStatistics)
{
  constexpr double eps = 1.e-8;
  PITTS::internal::TimingStatistics timings;

  EXPECT_NEAR(0., timings.totalTime, eps);
  EXPECT_EQ(0, timings.calls);

  timings += std::chrono::minutes(1);

  EXPECT_NEAR(60., timings.totalTime, eps);
  EXPECT_EQ(1, timings.calls);
  EXPECT_NEAR(60., timings.minTime, eps);
  EXPECT_NEAR(60., timings.maxTime, eps);

  timings += std::chrono::seconds(22);

  EXPECT_NEAR(82., timings.totalTime, eps);
  EXPECT_EQ(2, timings.calls);
  EXPECT_NEAR(22., timings.minTime, eps);
  EXPECT_NEAR(60., timings.maxTime, eps);
}

TEST(PITTS_Timer, simple_function)
{
  simpleFunction(5, 7);
}

TEST(PITTS_Timer, simple_function_with_type)
{
  simpleFunctionWithType(5, 7);
}
