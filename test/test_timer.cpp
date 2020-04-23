#include <gtest/gtest.h>
#include "pitts_timer.hpp"

namespace
{
  void simpleFunction(int x, int y)
  {
    const auto timer = PITTS::timing::createScopedTimer();
    printf("x+y: %d\n", x+y);
  }

  void simpleFunctionWithType(int x, int y)
  {
    const auto timer = PITTS::timing::createScopedTimer<int>();
    printf("x+y: %d\n", x+y);
  }

  template<typename T>
  void simpleTemplateFunction(int x, int y)
  {
    const auto timer = PITTS::timing::createScopedTimer();
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
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleFunction(5, 7);

  ASSERT_EQ(1,  PITTS::timing::globalTimingStatisticsMap.size());
  for(auto& [scope, timings]: PITTS::timing::globalTimingStatisticsMap)
  {
    ASSERT_STREQ("simpleFunction", scope.function_name());
    ASSERT_STREQ("", scope.type_name());
    ASSERT_EQ(1, timings.calls);
  }
}

TEST(PITTS_Timer, simple_template_function)
{
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleTemplateFunction<int>(5, 7);

  ASSERT_EQ(1,  PITTS::timing::globalTimingStatisticsMap.size());
  for(auto& [scope, timings]: PITTS::timing::globalTimingStatisticsMap)
  {
    ASSERT_STREQ("simpleTemplateFunction<int>", scope.function_name());
    ASSERT_EQ(1, timings.calls);
  }
}

TEST(PITTS_Timer, simple_function_with_type)
{
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleFunctionWithType(5, 7);

  ASSERT_EQ(1,  PITTS::timing::globalTimingStatisticsMap.size());
  for(auto& [scope, timings]: PITTS::timing::globalTimingStatisticsMap)
  {
    ASSERT_STREQ("simpleFunctionWithType", scope.function_name());
    ASSERT_STREQ("<int>", scope.type_name());
    ASSERT_EQ(1, timings.calls);
  }
}
