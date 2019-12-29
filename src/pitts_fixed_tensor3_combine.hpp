/*! @file pitts_fixed_tensor3_combine.hpp
* @brief contract two simple fixed-dimension rank-3 tensors (along third and first dimension)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSOR3_COMBINE_HPP
#define PITTS_FIXED_TENSOR3_COMBINE_HPP

// includes
#include <array>
#include "pitts_fixed_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the result of multiplying two fixed-size rank-3 tensors (contraction of third and first dimension)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N>
  auto combine(FixedTensor3<T,N>& t3a, FixedTensor3<T,N>& t3b)
  {
    if( t3a.r2() != t3b.r1() )
      throw std::invalid_argument("Dimension mismatch!");
    const auto r1 = t3a.r1();
    const auto r = t3a.r2();
    const auto r2 = t3b.r2();
    FixedTensor3<T,N*N> t3c(r1,r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < r2; j++)
      {
        for(int k1 = 0; k1 < N; k1++)
          for(int k2 = 0; k2 < N; k2++)
          {
            T tmp{};
            for(int l = 0; l < r; l++)
              tmp += t3a(i,k1,l)*t3b(l,k2,j);
            t3c(i,k2*N+k1,j) = tmp;
          }
      }
    return t3c;
  }

}


#endif // PITTS_FIXED_TENSOR3_COMBINE_HPP
