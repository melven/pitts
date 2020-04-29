#include <gtest/gtest.h>
#include "pitts_scope_info.hpp"

namespace
{
  constexpr auto simpleFunction(int, int)
  {
    auto scope = PITTS::internal::ScopeInfo::current();
    return scope;
  }

  constexpr auto templateFunction(std::vector<auto> x)
  {
    auto scope = PITTS::internal::ScopeInfo::current();
    return scope;
  }

  constexpr auto getHash = PITTS::internal::ScopeInfo::Hash();

  template<typename>
  struct TemplateType {};
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
  EXPECT_STREQ("simpleFunction", scope.function_name());

  // check that we can get the hash at compile time
  constexpr auto hash = getHash(scope);
}

TEST(PITTS_ScopeInfo, templateFunction)
{
  std::vector<double> bla = {1., 2., 3.};
  auto scope = templateFunction(bla);
  EXPECT_STREQ("templateFunction<double>", scope.function_name());

  // check that we can get the hash at compile time
  auto hash = getHash(scope);
}

TEST(PITTS_ScopeInfo, ArgumentInfo)
{
  constexpr PITTS::internal::ArgumentInfo noArgs;
  EXPECT_EQ("", noArgs.to_string());

  constexpr PITTS::internal::ArgumentInfo<2> arg2Info{{"x", "y"}, {1, 8}};
  EXPECT_EQ("x: 1, y: 8", arg2Info.to_string());

  EXPECT_NE(noArgs.hash_values(), arg2Info.hash_values());
}

TEST(PITTS_ScopeInfo, type)
{
  constexpr auto scopeNone = PITTS::internal::ScopeInfo::current();
  printf("type none: %s\n", scopeNone.type_name());
  EXPECT_STREQ("", scopeNone.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashNone = getHash(scopeNone);


  // int
  constexpr auto scopeInt = PITTS::internal::ScopeInfo::current<int>();
  printf("type int: %s\n", scopeInt.type_name());
  EXPECT_STREQ("<int>", scopeInt.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashInt = getHash(scopeInt);

  EXPECT_NE(getHash(scopeInt), getHash(scopeNone));
}
