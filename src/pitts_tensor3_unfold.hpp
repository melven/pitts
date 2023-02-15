/*! @file pitts_tensor3_unfold.hpp
* @brief Reshape a 3-dimensional tensor into a 2-dimensional tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-27
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensor3_unfold.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSOR3_UNFOLD)
import pitts_tensor3_unfold;
#define PITTS_TENSOR3_UNFOLD_HPP
#endif

// include guard
#ifndef PITTS_TENSOR3_UNFOLD_HPP
#define PITTS_TENSOR3_UNFOLD_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include "pitts_tensor3.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensor3_unfold;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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

  // explicit template instantiations
  template void unfold_left<float, Tensor2<float>>(const Tensor3<float>& t3, Tensor2<float>& mat);
  template void unfold_left<double, Tensor2<double>>(const Tensor3<double>& t3, Tensor2<double>& mat);
  template void unfold_right<float, Tensor2<float>>(const Tensor3<float>& t3, Tensor2<float>& mat);
  template void unfold_right<double, Tensor2<double>>(const Tensor3<double>& t3, Tensor2<double>& mat);
  template void unfold<float, std::vector<float>>(const Tensor3<float>& t3, std::vector<float>& v);
  template void unfold<double, std::vector<double>>(const Tensor3<double>& t3, std::vector<double>& v);
}
  
#endif // PITTS_TENSOR3_UNFOLD_HPP
