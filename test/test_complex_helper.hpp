// include guard
#ifndef TEST_COMPLEX_HELPER_HPP
#define TEST_COMPLEX_HELPER_HPP

// includes
#include <gtest/gtest.h>
#include <complex>

// ASSERT_NEAR for complex numbers
namespace testing
{
namespace internal
{

inline AssertionResult DoubleNearPredFormat(const char* expr1,
                                            const char* expr2,
                                            const char* abs_error_expr,
                                            const std::complex<double>& val1,
                                            const std::complex<double>& val2,
                                            double abs_error) {
  const auto diff = std::abs(val2-val1);
  if( diff <= abs_error ) return AssertionSuccess();

  return AssertionFailure()
      << "The difference between " << expr1 << " and " << expr2
      << " is\n" << diff << ",\nwhich exceeds " << abs_error_expr << ", where\n"
      << expr1 << " evaluates to\n" << val1 << ",\n"
      << expr2 << " evaluates to\n" << val2 << ",\nand\n"
      << abs_error_expr << " evaluates to " << abs_error << ".";
}

inline AssertionResult DoubleNearPredFormat(const char* expr1,
                                            const char* expr2,
                                            const char* abs_error_expr,
                                            const double& val1,
                                            const std::complex<double>& val2,
                                            double abs_error) {
  const auto diff = std::abs(val2-val1);
  if( diff <= abs_error ) return AssertionSuccess();

  return AssertionFailure()
      << "The difference between " << expr1 << " and " << expr2
      << " is\n" << diff << ",\nwhich exceeds " << abs_error_expr << ", where\n"
      << expr1 << " evaluates to\n" << val1 << ",\n"
      << expr2 << " evaluates to\n" << val2 << ",\nand\n"
      << abs_error_expr << " evaluates to " << abs_error << ".";
}
}
}

#endif /* TEST_COMPLEX_HELPER_HPP */
