// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_fixed_tensortrain_dot.hpp
* @brief inner products for simple tensor train format with fixed dimension
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
*
**/

// include guard
#ifndef PITTS_FIXED_TENSORTRAIN_DOT_HPP
#define PITTS_FIXED_TENSORTRAIN_DOT_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the inner product for two vectors in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N>
  T dot(const FixedTensorTrain<T, N>& TT1, const FixedTensorTrain<T, N>& TT2)
  {
    const auto timer = PITTS::timing::createScopedTimer<FixedTensorTrain<T,N>>();

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // We assume that the dimensions of the "|" is much larger than the "--",
    // so we contract "|" first and continue from the left to the right (like a zipper).
    //
    
    //double wtime = omp_get_wtime();
    double flops = 0;

    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);

    Tensor2<T> last_t2;
    for(int iSubTensor = 0; iSubTensor < TT1.nDims(); iSubTensor++)
    {
      const auto& subT1 = TT1.subTensors()[iSubTensor];
      const auto& subT2 = TT2.subTensors()[iSubTensor];
      const auto r11 = subT1.r1();
      const auto r12 = subT1.r2();
      const auto r21 = subT2.r1();
      const auto r22 = subT2.r2();
      const auto n = subT1.n();

      flops += 2.*r11*r12*r21*r22*(n+1.);

      std::swap(t2,last_t2);

      // prepare new result tensor for adding up
      t2.resize(r12,r22);
      for(int j = 0; j < r22; j++)
        for(int i = 0; i < r12; i++)
          t2(i,j) = T(0);

      for(int j = 0; j < r22; j++)
        for(int i = 0; i < r12; i++)
          for(int k = 0; k < n; k++)
            for(int j_ = 0; j_ < r21; j_++)
              for(int i_ = 0; i_ < r11; i_++)
                t2(i,j) += last_t2(i_,j_)*subT1(i_,k,i)*subT2(j_,k,j);
    }
    //wtime = omp_get_wtime()-wtime;
    //std::cout << "GFlop/s: " << flops/wtime*1.e-9 << std::endl;
    return t2(0,0);
  }

}


#endif // PITTS_FIXED_TENSORTRAIN_DOT_HPP
