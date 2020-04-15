#include <gtest/gtest.h>
#include "pitts_timer.hpp"

namespace
{
  void simpleFunction(int x, int y)
  {
    PITTS::timing::ScopedTimer timer;
    printf("x+y: %d\n", x+y);
  }
}

TEST(PITTS_Timer, simple_function)
{
  simpleFunction(5, 7);
}
