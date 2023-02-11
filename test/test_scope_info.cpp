#include <gtest/gtest.h>
#include "pitts_scope_info.hpp"

namespace
{
  constexpr auto simpleFunction(int, int)
  {
    auto scope = PITTS::internal::ScopeInfo::current();
    return scope;
  }

  template<typename T>
  constexpr auto templateFunction(std::vector<T> x)
  {
    auto scope = PITTS::internal::ScopeInfo::current();
    return scope;
  }

  constexpr auto getHash = PITTS::internal::ScopeInfo::Hash();

  template<typename>
  struct TemplateType {};
}

TEST(PITTS_ScopeInfo, simpleFunction)
{
  constexpr auto scope = simpleFunction(7, 8);
#ifdef __clang__
  EXPECT_STREQ("auto (anonymous namespace)::simpleFunction(int, int)", scope.function_name());
#else
  EXPECT_STREQ("constexpr auto {anonymous}::simpleFunction(int, int)", scope.function_name());
#endif

  // check that we can get the hash at compile time
  constexpr auto hash = getHash(scope);
}

TEST(PITTS_ScopeInfo, templateFunction)
{
  std::vector<double> bla = {1., 2., 3.};
  auto scope = templateFunction(bla);
#ifdef __clang__
  EXPECT_STREQ("auto (anonymous namespace)::templateFunction(std::vector<T>) [T = double]", scope.function_name());
#else
  EXPECT_STREQ("constexpr auto {anonymous}::templateFunction(std::vector<T>) [with T = double]", scope.function_name());
#endif

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
  std::cout << "type none: " << scopeNone.type_name() << "\n";
  EXPECT_EQ("", scopeNone.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashNone = getHash(scopeNone);


  // int
  constexpr auto scopeInt = PITTS::internal::ScopeInfo::current<int>();
  std::cout << "type int: " << scopeNone.type_name() << "\n";
  EXPECT_EQ("int", scopeInt.type_name());

  // check that we can get the hash at compile time
  constexpr auto hashInt = getHash(scopeInt);

  EXPECT_NE(getHash(scopeInt), getHash(scopeNone));
}

TEST(PITTS_ScopeInfo, ScopeWithArgumentInfo)
{
  constexpr auto scope = PITTS::internal::ScopeInfo::current();
  constexpr auto getHash = PITTS::internal::ScopeWithArgumentInfo::Hash();

  const auto scopeWithNoArgs = PITTS::internal::ScopeWithArgumentInfo{scope, {}};
  EXPECT_EQ("", scopeWithNoArgs.args.to_string());

  const auto scopeWithSingleArg = PITTS::internal::ScopeWithArgumentInfo{scope, {{"N"},{7}}};
  EXPECT_EQ("N: 7", scopeWithSingleArg.args.to_string());

  const auto scopeWithFourArgs = PITTS::internal::ScopeWithArgumentInfo{scope, {{"A", "B", "C", "D"},{1, 2, 3, 4}}};
  EXPECT_EQ("A: 1, B: 2, C: 3, D: 4", scopeWithFourArgs.args.to_string());

  EXPECT_NE(getHash(scopeWithNoArgs), getHash(scopeWithSingleArg));
  EXPECT_NE(getHash(scopeWithFourArgs), getHash(scopeWithSingleArg));
}
