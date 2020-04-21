#include <gtest/gtest.h>
#include "pitts_scope_info.hpp"

namespace
{
  constexpr auto simpleFunction(int, int)
  {
    constexpr PITTS::internal::ScopeInfo scope;
    return scope;
  }

  template<typename T>
  constexpr auto templateFunction(T x)
  {
    constexpr PITTS::internal::ScopeInfo scope;
    return scope;
  }
}

TEST(PITTS_ScopeInfo, internal_djb_hash)
{
  using PITTS::internal::djb_hash;
  EXPECT_EQ(djb_hash("hello"), djb_hash("hello"));
  EXPECT_NE(djb_hash("hello"), djb_hash("world"));
}

TEST(PITTS_ScopeInfo, simpleFunction)
{
  constexpr auto scope = simpleFunction(7, 8);
  EXPECT_EQ("simpleFunction", scope.function_name());

  // check that we can get the hash at compile time
  constexpr auto hash = scope.hash();
}

TEST(PITTS_ScopeInfo, templateFunction)
{
  constexpr auto scope = templateFunction(2.5);
  EXPECT_EQ("templateFunction<double>", scope.function_name());

  // check that we can get the hash at compile time
  constexpr auto hash = scope.hash();
}

TEST(PITTS_ScopeInfo, ArgumentInfo)
{
  constexpr PITTS::internal::ArgumentInfo noArgs;
  EXPECT_EQ("", noArgs.to_string());

  constexpr PITTS::internal::ArgumentInfo<2> arg2Info{{"x", "y"}, {1, 8}};
  EXPECT_EQ("x: 1, y: 8", arg2Info.to_string());
}

TEST(PITTS_ScopeInfo, type)
{
  constexpr PITTS::internal::ScopeInfo scopeNone;
  printf("type none: %s\n", scopeNone.type_name().data());
  EXPECT_EQ("", scopeNone.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashNone = scopeNone.hash();

  int dummy = 7;
  constexpr PITTS::internal::ScopeInfo scopeInt(dummy);
  printf("type int: %s\n", scopeInt.type_name().data());
  EXPECT_EQ("<int>", scopeInt.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashInt = scopeInt.hash();

  EXPECT_NE(scopeInt.hash(), scopeNone.hash());
}
