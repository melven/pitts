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
      for(int j = 0; j < r2; j++)
      {
        for(int i = 0; i < r2; i++)
        {
          T t2ij = T(0);
          // 4-way unrolling by hand
          int ij_;
          for(ij_ = 0; ij_ < r1*r1-3; ij_+=4)
          {
            const int i_0 = (ij_+0) % r1;
            const int j_0 = (ij_+0) / r1;
            const int i_1 = (ij_+1) % r1;
            const int j_1 = (ij_+1) / r1;
            const int i_2 = (ij_+2) % r1;
            const int j_2 = (ij_+2) / r1;
            const int i_3 = (ij_+3) % r1;
            const int j_3 = (ij_+3) / r1;

            Chunk<T> tmp0{};
            Chunk<T> tmp1{};
            Chunk<T> tmp2{};
            Chunk<T> tmp3{};
            for(int k = 0; k < nChunks; k++)
            {
              fmadd(subT.chunk(i_0,k,i), subT.chunk(j_0,k,j), tmp0);
              fmadd(subT.chunk(i_1,k,i), subT.chunk(j_1,k,j), tmp1);
              fmadd(subT.chunk(i_2,k,i), subT.chunk(j_2,k,j), tmp2);
              fmadd(subT.chunk(i_3,k,i), subT.chunk(j_3,k,j), tmp3);
            }
            t2ij += last_t2(i_0,j_0)*sum(tmp0);
            t2ij += last_t2(i_1,j_1)*sum(tmp1);
            t2ij += last_t2(i_2,j_2)*sum(tmp2);
            t2ij += last_t2(i_3,j_3)*sum(tmp3);
          }
          // remainder loop
          for(; ij_ < r1*r1; ij_++)
          {
            const int i_ = ij_ % r1;
            const int j_ = ij_ / r1;
            Chunk<T> tmp{};
            for(int k = 0; k < nChunks; k++)
              fmadd(subT.chunk(i_,k,i), subT.chunk(j_,k,j), tmp);
            t2ij += last_t2(i_,j_)*sum(tmp);
          }
          t2(i,j) = t2ij;
        }
      }
      last_t2 = t2;
    }
    return std::sqrt(last_t2(0,0));
  }

}


#endif // PITTS_TENSORTRAIN_NORM_HPP
