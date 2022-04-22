/*! @file pitts_tensor3_combine.hpp
* @brief contract two simple rank-3 tensors (along third and first dimension)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_COMBINE_HPP
#define PITTS_TENSOR3_COMBINE_HPP

// includes
#include <array>
#include "pitts_tensor3.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the result of multiplying two rank-3 tensors (contraction of third and first dimension)
  //!
  //! The resulting tensor is t3c with
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)
  //! with a tensor product of the second dimensions:
  //! * for swap=false, we use k=k2*n1+k1
  //! * for swap=true,  we use k=k2+n2*k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param t3a    first rank-3 tensor
  //! @param t3b    second rank-3 tensor
  //! @param swap   store second dimension in "transposed" order
  //! @return       resulting t3c (see formula above)
  //!
  template<typename T>
  auto combine(const Tensor3<T>& t3a, const Tensor3<T>& t3b, bool swap = false)
  {
    if( t3a.r2() != t3b.r1() )
      throw std::invalid_argument("Dimension mismatch!");

    // gather performance data
    const auto r1 = t3a.r1();
    const auto n1 = t3a.n();
    const auto r = t3a.r2();
    const auto n2 = t3b.n();
    const auto r2 = t3b.r2();
    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n1", "r", "n2", "r2", "swap"}, {r1, n1, r, n2, r2, int(swap)}},     // arguments
        {{r1*r2*r*n1*n2*kernel_info::FMA<T>()},    // flops
         {(r1*n1*r + r*n2*r2)*kernel_info::Load<T>() + r1*n2*r2*kernel_info::Store<T>()}}    // data transfers
        );

    Tensor3<T> t3c(r1,n1*n2,r2);
    std::unique_ptr<T[]> tmp{new T[n1*n2]};
    if( swap )
    {
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r1; i++)
        {
          for(int k1 = 0; k1 < n1; k1++)
            for(int k2 = 0; k2 < n2; k2++)
              tmp[k1*n2+k2] = T();
          for(int l = 0; l < r; l++)
            for(int k1 = 0; k1 < n1; k1++)
              for(int k2 = 0; k2 < n2; k2++)
                tmp[k1*n2+k2] += t3a(i,k1,l)*t3b(l,k2,j);

          for(int k1 = 0; k1 < n1; k1++)
            for(int k2 = 0; k2 < n2; k2++)
              t3c(i,k1*n2+k2,j) = tmp[k1*n2+k2];
        }
    }
    else // no swap
    {
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r1; i++)
        {
          for(int k1 = 0; k1 < n1; k1++)
            for(int k2 = 0; k2 < n2; k2++)
              tmp[k1*n2+k2] = T();
          for(int l = 0; l < r; l++)
            for(int k1 = 0; k1 < n1; k1++)
              for(int k2 = 0; k2 < n2; k2++)
                tmp[k1*n2+k2] += t3a(i,k1,l)*t3b(l,k2,j);

          for(int k1 = 0; k1 < n1; k1++)
            for(int k2 = 0; k2 < n2; k2++)
                t3c(i,k2*n1+k1,j) = tmp[k1*n2+k2];
        }
    }
    return t3c;
  }

}


#endif // PITTS_TENSOR3_COMBINE_HPP
