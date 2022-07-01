/*! @file pitts_tensor3_unfold.hpp
* @brief Reshape a 3-dimensional tensor into a 2-dimensional tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-27
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_UNFOLD_HPP
#define PITTS_TENSOR3_UNFOLD_HPP

// includes
#include "pitts_tensor3.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dispatch to either [i] or (i) for compatibility of std::vector and Eigen vectors
    template<typename T>
    auto& elem(T& v, int i)
    {
      constexpr bool has_bracket_operator = requires(const T& t, int i){t[i];};
      if constexpr (has_bracket_operator)
        return v[i];
      else
        return v(i);
    }
  }

  //! reshape a 3d tensor to a matrix combining the first two dimensions (r1*n x r2)
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //! @tparam MatrixType    class for the matrix, must support resize and (i,j) element-wise access
  //!
  //! @param t3     3d input tensor of dimension (r1,n,r2)
  //! @param mat    2d output tensor resized to dimension (r1*n,r2)
  //!
  template<typename T, class MatrixType>
  void unfold_left(const Tensor3<T>& t3, MatrixType& mat)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    mat.resize(r1*n, r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          mat(i+j*r1,k) = t3(i,j,k);
        }
  }


  //! reshape a 3d tensor to a matrix combining the last two dimensions (r1 x n*r2)
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //! @tparam MatrixType    class for the matrix, must support resize and (i,j) element-wise access
  //!
  //! @param t3     3d input tensor of dimension (r1,n,r2)
  //! @param mat    2d output tensor resized to dimension (r1,n*r2)
  //!
  template<typename T, class MatrixType>
  void unfold_right(const Tensor3<T>& t3, MatrixType& mat)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    mat.resize(r1, n*r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          mat(i,j+k*n) = t3(i,j,k);
        }
  }

  //! reshape a 3d tensor to a vector flattening dimensions
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //! @tparam VectorType    class for the vector, must support resize and (i) or [i] element-wise access
  //!
  //! @param t3     3d input tensor of dimension (r1,n,r2)
  //! @param vec    1d output tensor resized to dimension (r1*n*r2)
  //!
  template<typename T, class VectorType>
  void unfold(const Tensor3<T>& t3, VectorType& vec)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    vec.resize(r1*n*r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          internal::elem(vec, i+j*r1+k*n*r1) = t3(i,j,k);
        }
  }
}
  
#endif // PITTS_TENSOR3_UNFOLD_HPP
