/*! @file pitts_tensor3_fold.hpp
* @brief Reshape a 2-dimensional tensor into a 3-dimensional tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-27
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_FOLD_HPP
#define PITTS_TENSOR3_FOLD_HPP

// includes
#include <cassert>
#include <stdexcept>
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dispatch rows to either x.rows() or x.r1() (for compatibility with Eigen matrices and Tensor2)
    template<typename T>
    auto rows(const T& x)
    {
      constexpr bool has_r1 = requires(const T& t){t.r1();};
      if constexpr (has_r1)
        return x.r1();
      else
        return x.rows();
    }

    //! dispatch cols to either x.cols() or x.r2() (for compatibility with Eigen matrices and Tensor2)
    template<typename T>
    auto cols(const T& x)
    {
      constexpr bool has_r2 = requires(const T& t){t.r2();};
      if constexpr (has_r2)
        return x.r2();
      else
        return x.cols();
    }

    //! dispatch to either [i] or (i) for compatibility of std::vector and Eigen vectors
    template<typename T>
    auto elem(const T& v, int i)
    {
      constexpr bool has_bracket_operator = requires(const T& t, int i){t[i];};
      constexpr bool has_unary_paren_operator = requires(const T& t, int i){t(i);};
      constexpr bool has_binary_paren_operator = requires(const T& t, int i, int j){t(i, j);};
      static_assert(has_bracket_operator || has_unary_paren_operator || has_binary_paren_operator);
      if constexpr (has_bracket_operator)
        return v[i];
      else if constexpr (has_unary_paren_operator)
        return v(i);
      else //if constexpr (has_binary_paren_operator)
      {
        return v(i % rows(v), i / rows(v));
      }
    }
  }

  //! reshape a matrix to a 3d tensor splitting the first dimension (r1*n x r2)
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //! @tparam MatrixType    class for the matrix, must support (i,j) element-wise access and provide dimensions as r1(),r2(), or rows(),cols() member functions
  //!
  //! @param mat    input matrix of dimension (r1*n,r2)
  //! @param n      middle dimension of the output tensor, first dimension of mat must be a multiple of n
  //! @param t3     output tensor resized to dimensions (r1,n,r2)
  //!
  template<typename T, class MatrixType>
  void fold_left(const MatrixType& mat, int n, Tensor3<T>& t3)
  {
    assert(internal::rows(mat) % n == 0);
    const auto r1 = internal::rows(mat) / n;
    const auto r2 = internal::cols(mat);

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    t3.resize(r1, n, r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          t3(i,j,k) = mat(i+j*r1,k);
        }
  }


  //! reshape a matrix to a 3d tensor splitting the second dimension (r1 x n*r2)
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //! @tparam MatrixType    class for the matrix, must support (i,j) element-wise access and provide dimensions as r1(),r2(), or rows(),cols() member functions
  //!
  //! @param mat    input matrix of dimension (r1,n*r2)
  //! @param n      middle dimension of the output tensor, second dimension of mat must be a multiple of n
  //! @param t3     output tensor resized to dimensions (r1,n,r2)
  //!
  template<typename T, class MatrixType>
  void fold_right(const MatrixType& mat, int n, Tensor3<T>& t3)
  {
    const auto r1 = internal::rows(mat);
    assert(internal::cols(mat) % n == 0);
    const auto r2 = internal::cols(mat) / n;

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    t3.resize(r1, n, r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          t3(i,j,k) = mat(i,j+k*n);
        }
  }


  //! reshape a vector to a 3d tensor with given dimensions
  //!
  //! @tparam T           underlying data type (double, complex, ...)
  //! @tparam VectorType  class for the vector, must support (i) element-wise access and provide dimension as size()
  //!
  //! @param vec      input vector of dimension (r1*n*r2)
  //! @param r1       first dimension of the output tensor
  //! @param n        second dimension of the output tensor
  //! @param r2       third dimension of the output tensor
  //!
  template<typename T, class VectorType>
  void fold(const VectorType& v, int r1, int n, int r2, Tensor3<T>& t3)
  {
    assert(v.size() == r1*n*r2);

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    t3.resize(r1, n, r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          t3(i,j,k) = internal::elem(v, i+j*r1+k*n*r1);
        }
  }

  //! reshape a vector to a 3d tensor with given dimensions (without copying data)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param vec      input MultiVector of dimension (r1*n*r2, 1), moved from
  //! @param r1       first dimension of the output tensor
  //! @param n        second dimension of the output tensor
  //! @param r2       third dimension of the output tensor
  //! @return         resulting Tensor3
  //!
  template<typename T>
  Tensor3<T> fold(MultiVector<T>&& mv, int r1, int n, int r2)
  {
    if( r1*n*r2 != mv.rows() || mv.cols() != 1 )
      throw std::invalid_argument("Mismatching dimensions for MultiVector fold to Tensor3!");
    
    const auto reservedChunks = mv.reservedChunks();
    std::unique_ptr<Chunk<T>[]> data = std::move(mv);
    return Tensor3<T>(std::move(data), reservedChunks, r1, n, r2);
  }

  //! reshape a 2d tensor to a 3d tensor splitting the first dimension (r1*n x r2), without copying data
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //!
  //! @param mv     input tensor2 of dimension (r1*n,r2)
  //! @param n      middle dimension of the output tensor, first dimension of mat must be a multiple of n
  //! @param t3     output tensor resized to dimensions (r1,n,r2)
  //!
  template<typename T>
  Tensor3<T> fold_left(Tensor2<T>&& mv, long long n)
  {
    long long r1 = mv.r1() / n;
    long long r2 = mv.r2();
    assert(mv.r1() % n == 0);
    
    const auto reservedChunks = mv.reservedChunks();
    std::unique_ptr<Chunk<T>[]> data = std::move(mv);
    return Tensor3<T>(std::move(data), reservedChunks, r1, n, r2);
  }

  //! reshape a 2d tensor to a 3d tensor splitting the second dimension (r1 x n*r2), without copying data
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //!
  //! @param mv     input tensor2 of dimension (r1,n*r2)
  //! @param n      middle dimension of the output tensor, first dimension of mat must be a multiple of n
  //! @param t3     output tensor resized to dimensions (r1,n,r2)
  //!
  template<typename T>
  Tensor3<T> fold_right(Tensor2<T>&& mv, long long n)
  {
    long long r1 = mv.r1();
    long long r2 = mv.r2() / n;
    assert(mv.r2() % n == 0);
    
    const auto reservedChunks = mv.reservedChunks();
    std::unique_ptr<Chunk<T>[]> data = std::move(mv);
    return Tensor3<T>(std::move(data), reservedChunks, r1, n, r2);
  }

}
  
#endif // PITTS_TENSOR3_FOLD_HPP
