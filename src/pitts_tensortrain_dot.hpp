/*! @file pitts_tensortrain_dot.hpp
* @brief inner products for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_DOT_HPP
#define PITTS_TENSORTRAIN_DOT_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the inner product for two vectors in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T dot(const TensorTrain<T>& TT1, const TensorTrain<T>& TT2)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    if( TT1.dimensions() != TT2.dimensions() )
      throw std::invalid_argument("TensorTrain dot dimensions mismatch!");

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // Algorithm starts on the right and works like a zipper...
    //
    
    //double wtime = omp_get_wtime();
    double flops = 0;

    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);
    Tensor3<T> t3;

    // iterate from left to right
    const int nDim = TT1.subTensors().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subT1 = TT1.subTensors()[iDim];
      const auto& subT2 = TT2.subTensors()[iDim];
      const auto r11 = subT1.r1();
      const auto r12 = subT1.r2();
      const auto r21 = subT2.r1();
      const auto r22 = subT2.r2();
      const auto n = subT1.n();

      // first contraction: subT2(:,:,*) * t2(:,*)
      t3.resize(r21, n, r12);
      for(int i = 0; i < r21; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r12; k++)
          {
            T tmp{};
            for(int l = 0; l < r22; l++)
              tmp += subT2(i,j,l) * t2(k,l);
            t3(i,j,k) = tmp;
          }

      // second contraction: subT1(:,*,*) * t3(:,*,*)
      t2.resize(r11,r21);
      for(int i = 0; i < r11; i++)
        for(int j = 0; j < r21; j++)
        {
          T tmp{};
          for(int k = 0; k < n; k++)
            for(int l = 0; l < r12; l++)
              tmp += subT1(i,k,l) * t3(j,k,l);
          t2(i,j) = tmp;
        }
    }
    return t2(0,0);
  }

}


#endif // PITTS_TENSORTRAIN_DOT_HPP
