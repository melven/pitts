// helper functionality for tests involving Eigen Arrays

// include guard
#ifndef EIGEN_TEST_HELPER_HPP
#define EIGEN_TEST_HELPER_HPP

// includes
#include <gtest/gtest.h>
#include <Eigen/Dense>


// allow ASSERT_NEAR with Eigen classes
namespace testing
{
namespace internal
{
template<typename Derived1, typename Derived2>
AssertionResult DoubleNearPredFormat(const char* expr1,
                                     const char* expr2,
                                     const char* abs_error_expr,
                                     const Eigen::DenseBase<Derived1>& val1,
                                     const Eigen::DenseBase<Derived2>& val2,
                                     double abs_error) {
  // evaluate expression templates
  auto val1_ = val1.eval();
  auto val2_ = val2.eval();

  // deal with special case of empty matrices (which are always equal)
  if ( (val1_.rows() * val1_.cols() == 0) &&
       (val2_.rows() * val2_.cols() == 0) &&
       (val1_.rows() == val2_.rows() ) &&
       (val1_.cols() == val2_.cols() ) )
    return AssertionSuccess();

  // handle different shapes
  if( val1_.rows() != val2_.rows() ||
      val1_.cols() != val2_.cols() )
    return AssertionFailure()
      << "Mismatching dimensions between " << expr1 << " and " << expr2 << ", where\n"
      << expr1 << " evaluates to\n" << val1 << ",\nand\n"
      << expr2 << " evaluates to\n" << val2 << ".";

  // let Eigen determine if they are equal enough
  auto diff = (val1_-val2_).array().abs().eval();
  if (diff.maxCoeff() <= abs_error) return AssertionSuccess();

  // TODO(wan): do not print the value of an expression if it's
  // already a literal.
  return AssertionFailure()
      << "The difference between " << expr1 << " and " << expr2
      << " is\n" << diff << ",\nwhich exceeds " << abs_error_expr << ", where\n"
      << expr1 << " evaluates to\n" << val1 << ",\n"
      << expr2 << " evaluates to\n" << val2 << ",\nand\n"
      << abs_error_expr << " evaluates to " << abs_error << ".";
}
}
}

#endif // EIGEN_TEST_HELPER_HPP
