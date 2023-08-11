// helper functionality for tests involving Eigen Arrays

// include guard
#ifndef EIGEN_TEST_HELPER_HPP
#define EIGEN_TEST_HELPER_HPP

// includes
#include <gtest/gtest.h>
#include "pitts_eigen.hpp"
#include <random>


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

namespace
{
  // helper function to generate random orthogonal matrix
  Eigen::MatrixXd randomOrthoMatrix(int n, int m)
  {
    // Uses the formula X(X^TX)^(-1/2) with X_ij drawn from the normal distribution N(0,1)
    // Source theorem 2.2.1 from
    // Y. Chikuse: "Statistics on Special Manifolds", Springer, 2003
    // DOI: 10.1007/978-0-387-21540-2
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::normal_distribution<> distribution(0,1);

    Eigen::MatrixXd X(n,m);
    for(int i = 0; i < n; i++)
      for(int j = 0; j < m; j++)
        X(i,j) = distribution(randomGenerator);

    // calculate the SVD X = U S V^T
#if EIGEN_VERSION_AT_LEAST(3,4,90)
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinV | Eigen::ComputeThinU> svd(X);
#else
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif
    // now we can robustly calculate X(X^TX)^(-1/2) = U S V^T ( V S U^T U S V^T )^(-1/2) = U S V^T ( V S^2 V^T )^(-1/2) = U S V^T ( V S^(-1) V^T ) = U V^T
    return svd.matrixU() * svd.matrixV().transpose();
  }
}

#endif // EIGEN_TEST_HELPER_HPP
