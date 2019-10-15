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
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_tensortrain.hpp"

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
    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // We assume that the dimensions of the "|" is much larger than the "--",
    // so we contract "|" first and continue from the left to the right (like a zipper).
    //
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2;
    Tensor2<T> last_t2(1,1);
    last_t2(0,0) = T(1);
    for(int iSubTensor = 0; iSubTensor < TT1.dimensions.size(); iSubTensor++)
    {
      const auto& subT1 = TT1.subTensors()[iSubTensor];
      const auto& subT2 = TT2.subTensors()[iSubTensor];
      const auto r11 = subT1.r1();
      const auto r12 = subT1.r2();
      const auto r21 = subT2.r1();
      const auto r22 = subT2.r2();
      const auto nChunks = subT1.nChunks();
      t2.resize(r12,r22);
      for(int i = 0; i < r12; i++)
      {
        for(int j = 0; j < r22; j++)
        {
          T t2ij = T(0);
          for(int i_ = 0; i_ < r11; i_++)
          {
            for(int j_ = 0; j_ < r21; j_++)
            {
              Chunk<T> tmp{};
              for(int k = 0; k < nChunks; k++)
                fmadd(subT1.chunk(i_,k,i), subT2.chunk(j_,k,j), tmp);
              t2ij += last_t2(i_,j_)*sum(tmp);
            }
          }
          t2(i,j) = t2ij;
        }
      }
      last_t2 = t2;
    }
    return last_t2(0,0);
  }

}


#endif // PITTS_TENSORTRAIN_DOT_HPP
