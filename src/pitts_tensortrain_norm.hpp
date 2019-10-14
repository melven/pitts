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
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_tensortrain.hpp"

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
    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // where the top and the bottom are the same tensor.
    // We assume that the dimensions of the "|" is much larger than the "--",
    // so we contract "|" first and continue from the left to the right (like a zipper).
    //
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2;
    Tensor2<T> last_t2(1,1);
    last_t2(0,0) = T(1);
    for(const auto& subT: TT.subTensors())
    {
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto nChunks = subT.nChunks();
      t2.resize(r2,r2);
      for(int i = 0; i < r2; i++)
      {
        for(int j = 0; j < r2; j++)
        {
          t2(i,j) = T(0);
          for(int i_ = 0; i_ < r1; i_++)
          {
            for(int j_ = 0; j_ < r1; j_++)
            {
              Chunk<T> tmp{};
              for(int k = 0; k < nChunks; k++)
                fmadd(subT.chunk(i_,k,i), subT.chunk(j_,k,j), tmp);
              t2(i,j) += last_t2(i_,j_)*sum(tmp);
            }
          }
        }
      }
      last_t2 = t2;
    }
    return std::sqrt(last_t2(0,0));
  }

}


#endif // PITTS_TENSORTRAIN_NORM_HPP
