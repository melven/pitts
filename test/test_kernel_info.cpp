#include <gtest/gtest.h>
#include "pitts_kernel_info.hpp"
#include <complex>

namespace
{
  void check_equal(const PITTS::internal::Flops& a, const PITTS::internal::Flops& b, const double eps = 1.e-8)
  {
    EXPECT_EQ(a.noFMA, b.noFMA);
    EXPECT_NEAR(a.singlePrecision, b.singlePrecision, eps);
    EXPECT_NEAR(a.doublePrecision, b.doublePrecision, eps);
  }

  void check_equal(const PITTS::internal::Bytes& a, const PITTS::internal::Bytes& b, const double eps = 1.e-8)
  {
    EXPECT_NEAR(a.dataSize, b.dataSize, eps);
    EXPECT_NEAR(a.update, b.update, eps);
    EXPECT_NEAR(a.load, b.load, eps);
    EXPECT_NEAR(a.store, b.store, eps);
  }
}

TEST(PITTS_KernelInfo, Flops_addition)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  {
    constexpr Flops f = Flops{true, 1., 0.} + Flops{true, 2., 0.};
    EXPECT_EQ(true, f.noFMA);
    EXPECT_NEAR(3., f.singlePrecision, eps);
    EXPECT_NEAR(0., f.doublePrecision, eps);
  }

  {
    constexpr Flops f = Flops{true, 0., 3.} + Flops{false, 2., 2.};
    EXPECT_EQ(true, f.noFMA);
    EXPECT_NEAR(2., f.singlePrecision, eps);
    EXPECT_NEAR(5., f.doublePrecision, eps);
  }

  {
    constexpr Flops f = Flops{false, 0., 5.} + Flops{false, 0., 7.};
    EXPECT_EQ(false, f.noFMA);
    EXPECT_NEAR(0., f.singlePrecision, eps);
    EXPECT_NEAR(12., f.doublePrecision, eps);
  }
}

TEST(PITTS_KernelInfo, Flops_multiplication)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  {
    constexpr Flops f = 3*Flops{true, 1., 0.};
    EXPECT_EQ(true, f.noFMA);
    EXPECT_NEAR(3., f.singlePrecision, eps);
    EXPECT_NEAR(0., f.doublePrecision, eps);
  }

  {
    constexpr Flops f = 0.5*Flops{true, 4., 10.};
    EXPECT_EQ(true, f.noFMA);
    EXPECT_NEAR(2., f.singlePrecision, eps);
    EXPECT_NEAR(5., f.doublePrecision, eps);
  }

  {
    constexpr Flops f = 4*Flops{false, 0., 3.};
    EXPECT_EQ(false, f.noFMA);
    EXPECT_NEAR(0., f.singlePrecision, eps);
    EXPECT_NEAR(12., f.doublePrecision, eps);
  }
}

TEST(PITTS_KernelInfo, BasicFlops_predefs)
{
  using namespace PITTS::internal;

  check_equal(Flops{true, 1., 0.}, Add<float>());
  check_equal(Flops{true, 1., 0.}, Mult<float>());
  check_equal(Flops{false, 2., 0.}, FMA<float>());

  check_equal(Flops{true, 0., 1.}, Add<double>());
  check_equal(Flops{true, 0., 1.}, Mult<double>());
  check_equal(Flops{false, 0, 2}, FMA<double>());

  check_equal(Flops{false, 0, 0}, NoOp<double>());

  check_equal(2*Add<float>(), Add<std::complex<float>>());
  check_equal(4*Mult<float>()+2*Add<float>(), Mult<std::complex<float>>());
  check_equal(4*FMA<float>(), FMA<std::complex<float>>());

  check_equal(2*Add<double>(), Add<std::complex<double>>());
  check_equal(4*Mult<double>()+2*Add<double>(), Mult<std::complex<double>>());
  check_equal(4*FMA<double>(), FMA<std::complex<double>>());
}

TEST(PITTS_KernelInfo, Bytes_addition)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  constexpr Bytes b = Bytes{1., 10., 100., 1000.} + Bytes{3., 0.3, 0.03, 0.003};
  EXPECT_NEAR(4., b.dataSize, eps);
  EXPECT_NEAR(10.3, b.update, eps);
  EXPECT_NEAR(100.03, b.load, eps);
  EXPECT_NEAR(1000.003, b.store, eps);
}

TEST(PITTS_KernelInfo, Bytes_multiplication)
{
  constexpr double eps = 1.e-8;
  using namespace PITTS::internal;

  constexpr Bytes b = 0.5*Bytes{2., 3., 5., 7.};
  EXPECT_NEAR(1., b.dataSize, eps);
  EXPECT_NEAR(1.5, b.update, eps);
  EXPECT_NEAR(2.5, b.load, eps);
  EXPECT_NEAR(3.5, b.store, eps);
}

TEST(PITTS_KernelInfo, BasicBytes_predefs)
{
  using namespace PITTS::internal;

  check_equal(Bytes{4., 4., 0., 0.}, Update<float>());
  check_equal(Bytes{8., 8., 0., 0.}, Update<double>());
  check_equal(Bytes{8., 8., 0., 0.}, Update<std::complex<float>>());
  check_equal(Bytes{16., 16., 0., 0.}, Update<std::complex<double>>());

  check_equal(Bytes{4., 0., 4., 0.}, Load<float>());
  check_equal(Bytes{8., 0., 8., 0.}, Load<double>());
  check_equal(Bytes{8., 0., 8., 0.}, Load<std::complex<float>>());
  check_equal(Bytes{16., 0., 16., 0.}, Load<std::complex<double>>());

  check_equal(Bytes{4., 0., 0., 4.}, Store<float>());
  check_equal(Bytes{8., 0., 0., 8.}, Store<double>());
  check_equal(Bytes{8., 0., 0., 8.}, Store<std::complex<float>>());
  check_equal(Bytes{16., 0., 0., 16.}, Store<std::complex<double>>());
}

TEST(PITTS_KernelInfo, KernelInfo_example)
{
  using namespace PITTS::kernel_info;

  // ddot operation with size n = 500;
  constexpr auto n = 500;
  using Type = double;
  constexpr auto info = KernelInfo{n*FMA<Type>(), 2*n*Load<Type>()};

  check_equal(PITTS::internal::Flops{false, 0., 500.*2}, info.flops);
  check_equal(PITTS::internal::Bytes{500.*2*8, 0., 500.*2*8, 0.}, info.bytes);
}
