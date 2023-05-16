/*! @file pitts_tensor3_impl.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_IMPL_HPP
#define PITTS_TENSOR3_IMPL_HPP

// includes
#include "pitts_tensor3.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement tensor3 copy
  template<typename T>
  void copy(const Tensor3<T>& a, Tensor3<T>& b)
  {
    const auto r1 = a.r1();
    const auto n = a.n();
    const auto r2 = a.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    b.resize(r1, n, r2);

#pragma omp parallel for collapse(3) schedule(static)
    for(int k = 0; k < r2; k++)
      for(long long j = 0; j < n; j++)
        for(int i = 0; i < r1; i++)
          b(i,j,k) = a(i,j,k);
  }
}


#endif // PITTS_TENSOR3_IMPL_HPP
