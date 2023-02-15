/*! @file pitts_fixed_tensor3_combine.hpp
* @brief contract two simple fixed-dimension rank-3 tensors (along third and first dimension)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_fixed_tensor3_combine.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_FIXED_TENSOR3_COMBINE)
import pitts_fixed_tensor3_combine;
#define PITTS_FIXED_TENSOR3_COMBINE_HPP
#endif

// include guard
#ifndef PITTS_FIXED_TENSOR3_COMBINE_HPP
#define PITTS_FIXED_TENSOR3_COMBINE_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <array>
#include "pitts_fixed_tensor3.hpp"
#include "pitts_performance.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_fixed_tensor3_combine;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! calculate the result of multiplying two fixed-size rank-3 tensors (contraction of third and first dimension)
  //!
  //! The resulting tensor is t3c with
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)
  //! with a tensor product of the second dimensions:
  //! * for swap=false, we use k=k2*N+k1
  //! * for swap=true,  we use k=k2+N*k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  //! @param t3a    first rank-3 tensor
  //! @param t3b    second rank-3 tensor
  //! @param swap   store second dimension in "transposed" order
  //! @return       resulting t3c (see formula above)
  //!
  template<typename T, int N>
  auto combine(const FixedTensor3<T,N>& t3a, const FixedTensor3<T,N>& t3b, bool swap = false)
  {
    if( t3a.r2() != t3b.r1() )
      throw std::invalid_argument("Dimension mismatch!");

    // gather performance data
    const auto r1 = t3a.r1();
    const auto r = t3a.r2();
    const auto r2 = t3b.r2();
    const auto timer = PITTS::performance::createScopedTimer<FixedTensor3<T,N>>(
        {{"r1", "r", "r2", "swap"}, {r1, r, r2, int(swap)}},     // arguments
        {{r1*r2*r*N*N*kernel_info::FMA<T>()},    // flops
         {(r1*N*r + r*N*r2)*kernel_info::Load<T>() + r1*N*r2*kernel_info::Store<T>()}}    // data transfers
        );

    FixedTensor3<T,N*N> t3c(r1,r2);
    if( swap )
    {
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r1; i++)
        {
          T tmp[N][N];
          for(int k1 = 0; k1 < N; k1++)
            for(int k2 = 0; k2 < N; k2++)
              tmp[k1][k2] = T(0);
          for(int l = 0; l < r; l++)
            for(int k1 = 0; k1 < N; k1++)
              for(int k2 = 0; k2 < N; k2++)
                tmp[k1][k2] += t3a(i,k1,l)*t3b(l,k2,j);

          for(int k1 = 0; k1 < N; k1++)
            for(int k2 = 0; k2 < N; k2++)
              t3c(i,k1*N+k2,j) = tmp[k1][k2];
        }
    }
    else // no swap
    {
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r1; i++)
        {
          T tmp[N][N];
          for(int k1 = 0; k1 < N; k1++)
            for(int k2 = 0; k2 < N; k2++)
              tmp[k1][k2] = T(0);
          for(int l = 0; l < r; l++)
            for(int k1 = 0; k1 < N; k1++)
              for(int k2 = 0; k2 < N; k2++)
                tmp[k1][k2] += t3a(i,k1,l)*t3b(l,k2,j);

          for(int k1 = 0; k1 < N; k1++)
            for(int k2 = 0; k2 < N; k2++)
                t3c(i,k2*N+k1,j) = tmp[k1][k2];
        }
    }
    return t3c;
  }

  // explicit template instantiations
}


#endif // PITTS_FIXED_TENSOR3_COMBINE_HPP
