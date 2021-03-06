#include <gtest/gtest.h>
#include "pitts_type_name.hpp"


class TestClassName;

TEST(PITTS_TypeName, simple_types)
{
  using PITTS::internal::TypeName;
  //using namespace PITTS::internal;

  ASSERT_EQ("int", TypeName::name<int>());
  ASSERT_EQ("const int", TypeName::name<const int>());
  ASSERT_EQ("const char*", TypeName::name<const char*>());

  ASSERT_EQ("TestClassName", TypeName::name<TestClassName>());
}

TEST(PITTS_TypeName, compile_time)
{
  using PITTS::internal::TypeName;

  // ensure this runs at compile time...
  static constexpr auto dummy = TypeName::name<unsigned int&>();
  ASSERT_EQ("unsigned int&", dummy);
}
