/*! @file pitts_tensortrain_norm.hpp
* @brief norms for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-09
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORM_HPP
#define PITTS_TENSORTRAIN_NORM_HPP

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
  //! calculate the 2-norm for a vector in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T norm2(const TensorTrain<T>& TT)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // where the top and the bottom are the same tensor.
    // Algorithm starts on the right and works like a zipper...
    //
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);
    Tensor3<T> t3;

    // iterate from right to left
    const int nDim = TT.subTensors().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subT = TT.subTensors()[iDim];
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();

      // first contraction: subT(:,:,*) * t2(:,*)
      t3.resize(r1, n, r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
          {
            T tmp{};
            for(int l = 0; l < r2; l++)
              tmp += subT(i,j,l) * t2(k,l);
            t3(i,j,k) = tmp;
          }

      // second contraction: subT(:,*,*) * t3(:,*,*)
      t2.resize(r1,r1);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r1; j++)
        {
          T tmp{};
          for(int k = 0; k < n; k++)
            for(int l = 0; l < r2; l++)
              tmp += subT(i,k,l) * t3(j,k,l);
          t2(i,j) = tmp;
        }
    }

    return std::sqrt(t2(0,0));
  }

}


#endif // PITTS_TENSORTRAIN_NORM_HPP
