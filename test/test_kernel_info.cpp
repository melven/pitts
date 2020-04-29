#include <gtest/gtest.h>
#include "pitts_kernel_info.hpp"

namespace
{
  void check_equal(const PITTS::internal::Flops& a, const PITTS::internal::Flops& b, const double eps = 1.e-8)
  {
    EXPECT_EQ(a.noFMA, b.noFMA);
    EXPECT_NEAR(a.singlePrecision, b.singlePrecision, eps);
    EXPECT_NEAR(a.doublePrecision, b.doublePrecision, eps);
  }
}

TEST(PITTS_KernelInfo, Flops_addition)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  Flops f = Flops{true, 1., 0.} + Flops{true, 2., 0.};
  EXPECT_EQ(true, f.noFMA);
  EXPECT_NEAR(3., f.singlePrecision, eps);
  EXPECT_NEAR(0., f.doublePrecision, eps);

  f = Flops{true, 0., 3.} + Flops{false, 2., 2.};
  EXPECT_EQ(true, f.noFMA);
  EXPECT_NEAR(2., f.singlePrecision, eps);
  EXPECT_NEAR(5., f.doublePrecision, eps);

  f = Flops{false, 0., 5.} + Flops{false, 0., 7.};
  EXPECT_EQ(false, f.noFMA);
  EXPECT_NEAR(0., f.singlePrecision, eps);
  EXPECT_NEAR(12., f.doublePrecision, eps);
}

TEST(PITTS_KernelInfo, Flops_multiplication)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  Flops f = 3*Flops{true, 1., 0.};
  EXPECT_EQ(true, f.noFMA);
  EXPECT_NEAR(3., f.singlePrecision, eps);
  EXPECT_NEAR(0., f.doublePrecision, eps);

  f = 0.5*Flops{true, 4., 10.};
  EXPECT_EQ(true, f.noFMA);
  EXPECT_NEAR(2., f.singlePrecision, eps);
  EXPECT_NEAR(5., f.doublePrecision, eps);

  f = 4*Flops{false, 0., 3.};
  EXPECT_EQ(false, f.noFMA);
  EXPECT_NEAR(0., f.singlePrecision, eps);
  EXPECT_NEAR(12., f.doublePrecision, eps);
}

TEST(PITTS_KernelInfo, BasicFlops_predefs)
{
  using namespace PITTS::internal;

  check_equal(Flops{false, 1., 0.}, Add<float>());
  check_equal(Flops{false, 1., 0.}, Mult<float>());
  check_equal(Flops{true, 2., 0.}, FMA<float>());

  check_equal(Flops{false, 0., 1.}, Add<double>());
  check_equal(Flops{false, 0., 1.}, Mult<double>());
  check_equal(Flops{true, 0, 2}, FMA<double>());

  check_equal(2*Add<float>(), Add<std::complex<float>>());
  check_equal(4*Mult<float>()+2*Add<float>(), Mult<std::complex<float>>());
  check_equal(4*FMA<float>(), FMA<std::complex<float>>());

  check_equal(2*Add<double>(), Add<std::complex<double>>());
  check_equal(4*Mult<double>()+2*Add<double>(), Mult<std::complex<double>>());
  check_equal(4*FMA<double>(), FMA<std::complex<double>>());
}

