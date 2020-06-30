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

    Tensor2<T> last_t2(1,1);
    for(int iSubTensor = 0; iSubTensor < TT1.dimensions().size(); iSubTensor++)
    {
      const auto& subT1 = TT1.subTensors()[iSubTensor];
      const auto& subT2 = TT2.subTensors()[iSubTensor];
      const auto r11 = subT1.r1();
      const auto r12 = subT1.r2();
      const auto r21 = subT2.r1();
      const auto r22 = subT2.r2();
      const auto n = subT1.n();
      const auto nChunks = subT1.nChunks();

      flops += 2.*r11*r12*r21*r22*(n+1.);

      // loop unrolling parameter (but unrolling is done by hand below!)
      constexpr auto unrollSize = 2;

      // copy last result
      const int r11_padded = unrollSize * (1 + (r11-1)/unrollSize);
      const int r21_padded = unrollSize * (1 + (r21-1)/unrollSize);
      last_t2.resize(r11_padded,r21_padded);
      for(int j_ = 0; j_ < r21_padded; j_++)
        for(int i_ = 0; i_ < r11_padded; i_++)
          last_t2(i_,j_) = (i_ < r11 && j_ < r21) ? t2(i_,j_) : T(0);

      // prepare new result tensor for adding up
      t2.resize(r12,r22);
      for(int j = 0; j < r22; j++)
        for(int i = 0; i < r12; i++)
          t2(i,j) = T(0);
      T* t2data = &t2(0,0);
      const auto t2size = r12*r22;

#pragma omp parallel for schedule(static) reduction(+:t2data[:t2size])
      for(int k = 0; k < nChunks; k++)
      {
        for(int j = 0; j < r22; j++)
          for(int i = 0; i < r12; i++)
          {
            Chunk<T> tmp1{};
            for(int jb_ = 0; jb_ < r21; jb_+=2)
            {
              // 2-way unrolling (unrollSize == 2)
              Chunk<T> tmp20{};
              Chunk<T> tmp21{};
              for(int i_ = 0; i_ < r11; i_++)
              {
                fmadd(last_t2(i_,jb_+0),subT1.chunk(i_,k,i),tmp20);
                fmadd(last_t2(i_,jb_+1),subT1.chunk(i_,k,i),tmp21);
              }
              fmadd(subT2.chunk(jb_,k,j),tmp20,tmp1);
              if( jb_+1 < r21 )
                fmadd(subT2.chunk(jb_+1,k,j),tmp21,tmp1);
            }
            // this directly works on a pointer for the data of t2 to allow an OpenMP array reduction
            t2data[i+j*r12] += sum(tmp1);
          }
      }
    }
    //wtime = omp_get_wtime()-wtime;
    //std::cout << "GFlop/s: " << flops/wtime*1.e-9 << std::endl;
    return t2(0,0);
  }

}


#endif // PITTS_TENSORTRAIN_DOT_HPP
