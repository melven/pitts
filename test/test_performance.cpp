#include <gtest/gtest.h>
#include "pitts_performance.hpp"

namespace
{
  using namespace PITTS;
  void simpleFunction(int x, int y)
  {
    const auto timer = PITTS::performance::createScopedTimer({{"x","y"},{x,y}}, {});
    printf("x+y: %d\n", x+y);
  }

  void simpleFunctionWithType(int x, int y)
  {
    const auto timer = PITTS::performance::createScopedTimer<int>({{"x","y"},{x,y}},{200*kernel_info::FMA<double>(), 20*kernel_info::Update<double>()});
    printf("x+y: %d\n", x+y);
  }

  template<typename T>
  void simpleTemplateFunction(int x, int y)
  {
    const auto timer = PITTS::performance::createScopedTimer({{"x","y"},{x,y}},{10*kernel_info::Add<float>(), 10*kernel_info::Load<float>()});
    printf("x+y: %d\n", x+y);
  }
}


TEST(PITTS_Performance, simple_function)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  simpleFunction(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
    ASSERT_STREQ("simpleFunction", scopeWithArgs.scope.function_name());
    ASSERT_STREQ("", scopeWithArgs.scope.type_name());
    ASSERT_EQ(1, performance.timings.calls);
  }

  // just to see the result
  PITTS::performance::printStatistics();
}

TEST(PITTS_Performance, simple_template_function)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  simpleTemplateFunction<int>(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
    ASSERT_STREQ("simpleTemplateFunction<int>", scopeWithArgs.scope.function_name());
    ASSERT_EQ(1, performance.timings.calls);
  }

  simpleTemplateFunction<double>(5, 7);
  ASSERT_EQ(2,  PITTS::performance::globalPerformanceStatisticsMap.size());

  // just to see the result
  PITTS::performance::printStatistics();
}

TEST(PITTS_Performance, simple_function_with_type)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  simpleFunctionWithType(5, 7);
  simpleFunctionWithType(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
    ASSERT_STREQ("simpleFunctionWithType", scopeWithArgs.scope.function_name());
    ASSERT_STREQ("<int>", scopeWithArgs.scope.type_name());
    ASSERT_EQ(2, performance.timings.calls);
    ASSERT_EQ(400, performance.kernel.flops.doublePrecision);
    ASSERT_EQ(false, performance.kernel.flops.noFMA);
  }

  simpleFunctionWithType(5, 3);
  ASSERT_EQ(2,  PITTS::performance::globalPerformanceStatisticsMap.size());

  // just to see the result
  PITTS::performance::printStatistics();
}
