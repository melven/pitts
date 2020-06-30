#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <sstream>
#include "pitts_chunk_ops.hpp"

template<typename T>
class PITTS_ChunkOps: public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_CASE(PITTS_ChunkOps, TestTypes);


// helper functionality
namespace
{
  constexpr auto eps = 1.e-5;

  // random numbers
  template<typename T>
  void randomize(T &v)
  {
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    v = distribution(randomGenerator);
  }

  // specialize for complex numbers
  template<typename T>
  void randomize(std::complex<T>& v)
  {
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    v = std::complex<T>(distribution(randomGenerator), distribution(randomGenerator));
  }

  // random chunks
  template<typename T>
  void randomize(PITTS::Chunk<T>& chunk)
  {
    for(int i = 0; i < PITTS::Chunk<T>::size; i++)
      randomize(chunk[i]);
  }


  // nice output of arrays
  template<typename T, std::size_t N>
  std::string to_string(const std::array<T,N>& arr)
  {
    std::ostringstream out;
    out << "[";
    for(const auto& v: arr)
      out << " " << v;
    out << " ]";
    return out.str();
  }

}

// allow ASSERT_NEAR with Chunk
namespace testing
{
namespace internal
{
  template<typename T>
  AssertionResult DoubleNearPredFormat(const char* expr1,
                                       const char* expr2,
                                       const char* abs_error_expr,
                                       const PITTS::Chunk<T>& val1,
                                       const PITTS::Chunk<T>& val2,
                                       double abs_error)
  {
    // calculate difference
    std::array<double, PITTS::Chunk<T>::size> diff;
    for(int i = 0; i < PITTS::Chunk<T>::size; i++)
      diff[i] = (double) std::abs(val2[i] - val1[i]);

    for(int i = 0; i < PITTS::Chunk<T>::size; i++)
    {
      if( diff[i] > abs_error )
      {
        return AssertionFailure()
          << "The difference between " << expr1 << " and " << expr2
          << " is\n" << to_string(diff) << ",\nwhich exceeds " << abs_error_expr << ", where\n"
          << expr1 << " evaluated to\n" << to_string(val1) << ",\n"
          << expr2 << " evaluated to\n" << to_string(val2) << ",\n"
          << abs_error_expr << " evaluates to " << abs_error << ".";
      }
    }

    return AssertionSuccess();
  }
}
}


TYPED_TEST(PITTS_ChunkOps, fmadd)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a, b, c;
  randomize(a);
  randomize(b);
  randomize(c);

  const Chunk a_ref = a, b_ref = b, c_in = c;

  fmadd(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = a_ref[i]*b_ref[i] + c_in[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


TYPED_TEST(PITTS_ChunkOps, scalar_fmadd)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Type a;
  Chunk b, c;
  randomize(a);
  randomize(b);
  randomize(c);

  const Type a_ref = a;
  const Chunk b_ref = b, c_in = c;

  fmadd(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = a_ref*b_ref[i] + c_in[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


TYPED_TEST(PITTS_ChunkOps, fnmadd)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a, b, c;
  randomize(a);
  randomize(b);
  randomize(c);

  const Chunk a_ref = a, b_ref = b, c_in = c;

  fnmadd(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = -a_ref[i]*b_ref[i] + c_in[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


TYPED_TEST(PITTS_ChunkOps, sum)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a;
  randomize(a);

  const Chunk a_ref = a;

  const auto result = sum(a);

  Type result_ref{};
  for(int i = 0; i < Chunk::size; i++)
    result_ref += a[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_NEAR(std::real(result_ref), std::real(result), eps);
  EXPECT_NEAR(std::imag(result_ref), std::imag(result), eps);
}


TYPED_TEST(PITTS_ChunkOps, scaled_sum)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a;
  randomize(a);
  Type scale;
  randomize(scale);

  const Chunk a_ref = a;
  const Type scale_ref = scale;

  const auto result = scaled_sum(scale, a);

  Type result_ref{};
  for(int i = 0; i < Chunk::size; i++)
    result_ref += a[i];
  result_ref *= scale;

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(scale_ref, scale);
  EXPECT_NEAR(std::real(result_ref), std::real(result), eps);
  EXPECT_NEAR(std::imag(result_ref), std::imag(result), eps);
}


TYPED_TEST(PITTS_ChunkOps, negative_sign)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk v;
  randomize(v);

  const Chunk v_ref = v;

  Chunk nsgn;
  negative_sign(v, nsgn);

  for(int i = 0; i < Chunk::size; i++)
  {
    EXPECT_TRUE(std::real(nsgn[i]) * std::real(v[i]) <= 0);
    EXPECT_TRUE(std::imag(nsgn[i]) * std::imag(v[i]) <= 0);
  }
}


