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

  void functionWithTimerInLoop(int n)
  {
    for(int i = 0; i < n; i++)
    {
      const auto timer = PITTS::performance::createScopedTimer({}, {}, n);
    }
  }
}


TEST(PITTS_Performance, simple_function)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleFunction(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
#ifdef __clang__
    ASSERT_STREQ("void (anonymous namespace)::simpleFunction(int, int)", scopeWithArgs.scope.function_name());
#else
    ASSERT_STREQ("void {anonymous}::simpleFunction(int, int)", scopeWithArgs.scope.function_name());
#endif
    ASSERT_EQ("", scopeWithArgs.scope.type_name());
    ASSERT_EQ(1, performance.timings.calls);
  }

  // just to see the result
  PITTS::performance::printStatistics();
}

TEST(PITTS_Performance, simple_template_function)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleTemplateFunction<int>(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
#ifdef __clang__
    ASSERT_STREQ("void (anonymous namespace)::simpleTemplateFunction(int, int) [T = int]", scopeWithArgs.scope.function_name());
#else
    ASSERT_STREQ("void {anonymous}::simpleTemplateFunction(int, int) [with T = int]", scopeWithArgs.scope.function_name());
#endif
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
  PITTS::timing::globalTimingStatisticsMap.clear();

  simpleFunctionWithType(5, 7);
  simpleFunctionWithType(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(const auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
#ifdef __clang__
    ASSERT_STREQ("void (anonymous namespace)::simpleFunctionWithType(int, int)", scopeWithArgs.scope.function_name());
#else
    ASSERT_STREQ("void {anonymous}::simpleFunctionWithType(int, int)", scopeWithArgs.scope.function_name());
#endif
    ASSERT_EQ("int", scopeWithArgs.scope.type_name());
    ASSERT_EQ(2, performance.timings.calls);
    ASSERT_EQ(400, performance.kernel.flops.doublePrecision);
    ASSERT_EQ(false, performance.kernel.flops.noFMA);
  }

  simpleFunctionWithType(5, 3);
  ASSERT_EQ(2,  PITTS::performance::globalPerformanceStatisticsMap.size());

  // just to see the result
  PITTS::performance::printStatistics();
}

TEST(PITTS_Performance, combineTimingsPerFunction_simple)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  simpleFunctionWithType(5, 7);
  simpleFunctionWithType(5, 7);
  simpleFunctionWithType(5, 7);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  const auto timingStats = PITTS::internal::combineTimingsPerFunction(PITTS::performance::globalPerformanceStatisticsMap);
  ASSERT_EQ(1, timingStats.size());
  for(const auto& [scope, timings]: timingStats)
  {
    ASSERT_EQ(3, timings.calls);
  }
}

TEST(PITTS_Performance, combineTimingsPerFunction_differentArgs)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  simpleFunctionWithType(5, 7);
  simpleFunctionWithType(5, 4);
  simpleFunctionWithType(2, 7);

  ASSERT_EQ(3,  PITTS::performance::globalPerformanceStatisticsMap.size());
  const auto timingStats = PITTS::internal::combineTimingsPerFunction(PITTS::performance::globalPerformanceStatisticsMap);
  ASSERT_EQ(1, timingStats.size());
  for(const auto& [scope, timings]: timingStats)
  {
    ASSERT_EQ(3, timings.calls);
  }
}

TEST(PITTS_Performance, combineTimingsPerFunction_subScope)
{
  PITTS::performance::globalPerformanceStatisticsMap.clear();

  functionWithTimerInLoop(5);
  functionWithTimerInLoop(7);
  functionWithTimerInLoop(2);

  ASSERT_EQ(1,  PITTS::performance::globalPerformanceStatisticsMap.size());
  for(const auto& [scopeWithArgs, performance]: PITTS::performance::globalPerformanceStatisticsMap)
  {
    ASSERT_EQ(14, performance.timings.calls);
  }
  const auto timingStats = PITTS::internal::combineTimingsPerFunction(PITTS::performance::globalPerformanceStatisticsMap);
  ASSERT_EQ(1, timingStats.size());
  for(const auto& [scope, timings]: timingStats)
  {
    ASSERT_EQ(3, timings.calls);
  }
}

