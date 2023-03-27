#include <gtest/gtest.h>
#include <memory>
#include <complex>
#include <random>
#include <sstream>

//
// This file defines test cases for pitts_chunk_ops*.hpp
// It can be included in different test....cpp files for testing implementations for different hardware (e.g. AVX2 and AVX512)
//
#ifndef TEST_NAME
#error "Please define TEST_NAME before including this file!"
#endif
// one macro indirection, so the compiler expands TEST_NAME
#define MY_TYPED_TEST_HELPER(X,Y) TYPED_TEST(X,Y)
#define MY_TYPED_TEST(TEST_CASE) MY_TYPED_TEST_HELPER(TEST_NAME, TEST_CASE)

template<typename T>
class TEST_NAME: public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_CASE(TEST_NAME, TestTypes);


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


MY_TYPED_TEST(fmadd)
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


MY_TYPED_TEST(fmadd4)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a, b, c, d;
  randomize(a);
  randomize(b);
  randomize(c);
  randomize(d);

  const Chunk a_ref = a, b_ref = b, c_ref = c;

  fmadd(a, b, c, d);

  Chunk d_ref;
  for(int i = 0; i < Chunk::size; i++)
    d_ref[i] = a_ref[i]*b_ref[i] + c_ref[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_EQ(c_ref, c);
  EXPECT_NEAR(d_ref, d, eps);
}


MY_TYPED_TEST(scalar_fmadd)
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


MY_TYPED_TEST(scalar_mul)
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

  mul(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = a_ref*b_ref[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


MY_TYPED_TEST(mul)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a, b, c;
  randomize(a);
  randomize(b);
  randomize(c);

  const Chunk a_ref = a, b_ref = b, c_in = c;

  mul(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = a_ref[i]*b_ref[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


MY_TYPED_TEST(fnmadd)
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


MY_TYPED_TEST(fnmadd4)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a, b, c, d;
  randomize(a);
  randomize(b);
  randomize(c);
  randomize(d);

  const Chunk a_ref = a, b_ref = b, c_ref = c;

  fnmadd(a, b, c, d);

  Chunk d_ref;
  for(int i = 0; i < Chunk::size; i++)
    d_ref[i] = -a_ref[i]*b_ref[i] + c_ref[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_EQ(c_ref, c);
  EXPECT_NEAR(d_ref, d, eps);
}


MY_TYPED_TEST(scalar_fnmadd)
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

  fnmadd(a, b, c);

  Chunk c_ref;
  for(int i = 0; i < Chunk::size; i++)
    c_ref[i] = -a_ref*b_ref[i] + c_in[i];

  EXPECT_EQ(a_ref, a);
  EXPECT_EQ(b_ref, b);
  EXPECT_NEAR(c_ref, c, eps);
}


MY_TYPED_TEST(sum)
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


MY_TYPED_TEST(scaled_sum)
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


MY_TYPED_TEST(bcast_sum)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk a;
  randomize(a);

  const Chunk a_in = a;

  bcast_sum(a);

  Type result_ref{};
  for(int i = 0; i < Chunk::size; i++)
    result_ref += a_in[i];

  for(int i = 0; i < Chunk::size; i++)
  {
    EXPECT_NEAR(std::real(result_ref), std::real(a[i]), eps);
    EXPECT_NEAR(std::imag(result_ref), std::imag(a[i]), eps);
  }
}


MY_TYPED_TEST(index_bcast)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk src;
  randomize(src);
  Type value;
  randomize(value);

  const Chunk src_ref = src;
  const Type value_ref = value;

  for(int i = 0; i < Chunk::size+10; i++)
  {
    Chunk result;
    index_bcast(src, i, value, result);

    Chunk result_ref = result;
    if( i < Chunk::size )
      result_ref[i] = value;

    EXPECT_EQ(src, src_ref);
    EXPECT_EQ(value, value_ref);
    EXPECT_NEAR(result, result_ref, eps);
  }
}


MY_TYPED_TEST(masked_load_after)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk src;
  randomize(src);

  const Chunk src_ref = src;

  for(int i = 0; i < Chunk::size+10; i++)
  {
    Chunk result;
    masked_load_after(src, i, result);

    Chunk result_ref;
    for(int j = 0; j < i && j < Chunk::size; j++)
      result_ref[j] = Type(0);
    for(int j = i; j < Chunk::size; j++)
      result_ref[j] = src[j];

    EXPECT_EQ(src, src_ref);
    EXPECT_NEAR(result, result_ref, eps);
  }
}


MY_TYPED_TEST(masked_store_after)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk src;
  randomize(src);

  const Chunk src_ref = src;

  for(int i = 0; i < Chunk::size+10; i++)
  {
    Chunk result;
    randomize(result);
    const Chunk result_in = result;

    masked_store_after(src, i, result);

    Chunk result_ref;
    for(int j = 0; j < i && j < Chunk::size; j++)
      result_ref[j] = result_in[j];
    for(int j = i; j < Chunk::size; j++)
      result_ref[j] = src[j];

    EXPECT_EQ(src, src_ref);
    EXPECT_NEAR(result, result_ref, eps);
  }
}


MY_TYPED_TEST(unaligned_load)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  std::unique_ptr<Type[]> src(new Type[Chunk::size]);
  for(int i = 0; i < Chunk::size; i++)
    src[i] = Type(i + 0.7);

  std::unique_ptr<Type[]> src_ref(new Type[Chunk::size]);
  for(int i = 0; i < Chunk::size; i++)
    src_ref[i] = src[i];

  for(int i = 0; i < Chunk::size+10; i++)
  {
    Chunk result;
    randomize(result);

    const Type* pData = src.get();
    unaligned_load(pData, result);

    for(int i = 0; i < Chunk::size; i++)
    {
      EXPECT_EQ(src_ref[i], src[i]);
      EXPECT_EQ(src[i], result[i]);
    }
  }
}


MY_TYPED_TEST(unaligned_store)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  std::unique_ptr<Type[]> result(new Type[Chunk::size+10]);

  for(int iter = 0; iter < 10; iter++)
  {
    Chunk src;
    randomize(src);
    const Chunk src_ref = src;

    unaligned_store(src, result.get()+iter);

    for(int i = 0; i < Chunk::size; i++)
    {
      EXPECT_EQ(src_ref[i], src[i]);
      EXPECT_EQ(src[i], result[i+iter]);
    }
  }
}


MY_TYPED_TEST(streaming_store)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;

  Chunk src;
  randomize(src);

  const Chunk src_ref = src;

  for(int i = 0; i < Chunk::size+10; i++)
  {
    Chunk result;
    randomize(result);

    streaming_store(src, result);

    EXPECT_EQ(src_ref, src);
    EXPECT_EQ(src, result);
  }
}
