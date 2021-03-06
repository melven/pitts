/*! @file pitts_fixed_tensor3_apply.hpp
* @brief apply linear transformation to a simple fixed-dimension rank-3 tensor (actually a tensor contraction)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSOR3_APPLY_HPP
#define PITTS_FIXED_TENSOR3_APPLY_HPP

// includes
#include <array>
#include "pitts_fixed_tensor3.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! in-place calculate the result multiplying a fixed-size rank-3 tensor with a matrix (contraction of the second dimensions)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N, std::size_t N_=N>
  void apply(FixedTensor3<T,N>& t3, const std::array<std::array<T,N_>,N_>& M)
  {
    // gather performance data
    const auto r1 = t3.r1();
    const auto r2 = t3.r2();
    const auto timer = PITTS::performance::createScopedTimer<FixedTensor3<T,N>>(
        {{"r1", "r2"}, {r1, r2}},     // arguments
        {{r1*r2*N*N*kernel_info::FMA<T>()},    // flops
         {r1*r2*N*kernel_info::Update<T>() + N*N*kernel_info::Load<T>()}}    // data transfers
        );

    for(int i = 0; i < r1; i++)
      for(int j = 0; j < r2; j++)
      {
        T tmp[N]{};
        for(int k_ = 0; k_ < N; k_++)
          for(int k = 0; k < N; k++)
            tmp[k] += t3(i,k_,j)*M[k_][k];
        for(int k = 0; k < N; k++)
          t3(i,k,j) = tmp[k];
      }
  }

}


#endif // PITTS_FIXED_TENSOR3_APPLY_HPP
