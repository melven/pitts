#include <gtest/gtest.h>
#include "pitts_random.hpp"
#include <type_traits>

using namespace PITTS;

TEST(PITTS_Random, generateRandomSeed)
{
    const auto seed1 = internal::generateRandomSeed();
    const auto seed2 = internal::generateRandomSeed();
    const auto seed3 = internal::generateRandomSeed();
    EXPECT_NE(seed1, seed2);
    EXPECT_NE(seed1, seed3);
    EXPECT_NE(seed2, seed3);

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

    EXPECT_NE(val1, val2);
    EXPECT_NE(val1, val3);
    EXPECT_NE(val2, val3);
}

TEST(PITTS_Random, unitDistribution_float)
{
    internal::UniformUnitDistribution<float> distribution;

    const float f1 = distribution(internal::randomGenerator);
    const float f2 = distribution(internal::randomGenerator);
    const float f3 = distribution(internal::randomGenerator);

    EXPECT_NE(f1, f2);
    EXPECT_NE(f1, f3);
    EXPECT_NE(f2, f3);

    EXPECT_LE(std::abs(f1), 1);
    EXPECT_LE(std::abs(f2), 1);
    EXPECT_LE(std::abs(f3), 1);
}

TEST(PITTS_Random, unitDistribution_double)
{
    internal::UniformUnitDistribution<double> distribution;

    const double d1 = distribution(internal::randomGenerator);
    const double d2 = distribution(internal::randomGenerator);
    const double d3 = distribution(internal::randomGenerator);

    EXPECT_NE(d1, d2);
    EXPECT_NE(d1, d3);
    EXPECT_NE(d2, d3);

    EXPECT_LE(std::abs(d1), 1);
    EXPECT_LE(std::abs(d2), 1);
    EXPECT_LE(std::abs(d3), 1);
}

TEST(PITTS_Random, unitDistribution_complex)
{
    internal::UniformUnitDistribution<std::complex<float>> distribution;

    const std::complex<float> c1 = distribution(internal::randomGenerator);
    const std::complex<float> c2 = distribution(internal::randomGenerator);
    const std::complex<float> c3 = distribution(internal::randomGenerator);

    EXPECT_NE(c1, c2);
    EXPECT_NE(c1, c3);
    EXPECT_NE(c2, c3);

    EXPECT_LE(std::abs(c1), 1);
    EXPECT_LE(std::abs(c2), 1);
    EXPECT_LE(std::abs(c3), 1);
}

TEST(PITTS_Random, unitDistribution_double_complex)
{
    internal::UniformUnitDistribution<std::complex<double>> distribution;

    const std::complex<double> z1 = distribution(internal::randomGenerator);
    const std::complex<double> z2 = distribution(internal::randomGenerator);
    const std::complex<double> z3 = distribution(internal::randomGenerator);

    EXPECT_NE(z1, z2);
    EXPECT_NE(z1, z3);
    EXPECT_NE(z2, z3);

    EXPECT_LE(std::abs(z1), 1);
    EXPECT_LE(std::abs(z2), 1);
    EXPECT_LE(std::abs(z3), 1);
}

TEST(PITTS_Random, unitDistribution_discard_float)
{
    internal::UniformUnitDistribution<float> distribution;

    // copy generator
    auto generator = internal::randomGenerator;
    EXPECT_EQ(generator, internal::randomGenerator);

    float f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(1, generator);
    EXPECT_EQ(generator, internal::randomGenerator);

    for(int i = 0; i < 10; i++)
        f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(10, generator);
    EXPECT_EQ(generator, internal::randomGenerator);
}

TEST(PITTS_Random, unitDistribution_discard_double)
{
    internal::UniformUnitDistribution<double> distribution;

    // copy generator
    auto generator = internal::randomGenerator;
    EXPECT_EQ(generator, internal::randomGenerator);

    double f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(1, generator);
    EXPECT_EQ(generator, internal::randomGenerator);

    for(int i = 0; i < 10; i++)
        f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(10, generator);
    EXPECT_EQ(generator, internal::randomGenerator);
}

TEST(PITTS_Random, unitDistribution_discard_complex)
{
    internal::UniformUnitDistribution<std::complex<float>> distribution;

    // copy generator
    auto generator = internal::randomGenerator;
    EXPECT_EQ(generator, internal::randomGenerator);

    std::complex<float> f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(1, generator);
    EXPECT_EQ(generator, internal::randomGenerator);

    for(int i = 0; i < 10; i++)
        f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(10, generator);
    EXPECT_EQ(generator, internal::randomGenerator);
}

TEST(PITTS_Random, unitDistribution_discard_double_complex)
{
    internal::UniformUnitDistribution<std::complex<double>> distribution;

    // copy generator
    auto generator = internal::randomGenerator;
    EXPECT_EQ(generator, internal::randomGenerator);

    std::complex<double> f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(1, generator);
    EXPECT_EQ(generator, internal::randomGenerator);

    for(int i = 0; i < 10; i++)
        f = distribution(internal::randomGenerator);
    EXPECT_NE(generator, internal::randomGenerator);
    distribution.discard(10, generator);
    EXPECT_EQ(generator, internal::randomGenerator);
}
