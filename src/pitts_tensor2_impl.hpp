// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_impl.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
*
**/

// include guard
#ifndef PITTS_TENSOR2_IMPL_HPP
#define PITTS_TENSOR2_IMPL_HPP

// includes
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement tensor2 copy
  template<typename T>
  void copy(const Tensor2<T>& a, Tensor2<T>& b)
  {
    const auto r1 = a.r1();
    const auto r2 = a.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*r2*kernel_info::NoOp<T>()},    // flops
         {r1*r2*kernel_info::Store<T>()+r1*r2*kernel_info::Load<T>()}}  // data
        );


    b.resize(r1, r2);

#pragma omp parallel for collapse(2) schedule(static) if(r1*r2 > 500)
    for(long long j = 0; j < r2; j++)
      for(long long i = 0; i < r1; i++)
        b(i,j) = a(i,j);
  }
}


#endif // PITTS_TENSOR2_IMPL_HPP
