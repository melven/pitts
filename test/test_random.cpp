#include <gtest/gtest.h>
#include "pitts_random.hpp"
#include <type_traits>

using namespace PITTS;

TEST(PITTS_Random, generateRandomSeed)
{
    const auto seed1 = internal::generateRandomSeed();
    const auto seed2 = internal::generateRandomSeed();
    const auto seed3 = internal::generateRandomSeed();
    ASSERT_NE(seed1, seed2);
    ASSERT_NE(seed1, seed3);
    ASSERT_NE(seed2, seed3);

    static_assert(std::is_same_v<std::uint32_t, std::random_device::result_type>);
    static_assert(std::is_same_v<std::uint_fast64_t, decltype(internal::generateRandomSeed())>);
    static_assert(std::is_same_v<std::uint_fast64_t, decltype(internal::randomGenerator())>);
}

TEST(PITTS_Random, randomGenerator)
{
    // save old state
    const auto val1 = internal::randomGenerator();
    const auto val2 = internal::randomGenerator();
    const auto val3 = internal::randomGenerator();

    ASSERT_NE(val1, val2);
    ASSERT_NE(val1, val3);
    ASSERT_NE(val2, val3);
}