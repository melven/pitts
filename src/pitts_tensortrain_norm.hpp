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
    // We assume that the dimensions of the "|" is much larger than the "--",
    // so we contract "|" first and continue from the left to the right (like a zipper).
    //
    
    //double wtime = omp_get_wtime();
    double flops = 0;

    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);

    Tensor2<T> last_t2;
    for(const auto& subT: TT.subTensors())
    {
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();
      const auto nChunks = subT.nChunks();

      flops += 2.*r1*r1*(r2*(r2+1))/2.*(n+1.);

      // loop unrolling parameter (but unrolling is done by hand below!)
      constexpr auto unrollSize = 2;

      // copy last result
      const int r1_padded = unrollSize * (1 + (r1-1)/unrollSize);
      last_t2.resize(r1_padded,r1_padded);
      for(int j_ = 0; j_ < r1_padded; j_++)
        for(int i_ = 0; i_ < r1_padded; i_++)
          last_t2(i_,j_) = (i_ < r1 && j_ < r1) ? t2(i_,j_) : T(0);

      // prepare new result tensor for adding up
      t2.resize(r2,r2);
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r2; i++)
          t2(i,j) = T(0);
      T* t2data = &t2(0,0);
      const auto t2size = r2*r2;

#pragma omp parallel for schedule(static) reduction(+:t2data[:t2size])
      for(int k = 0; k < nChunks; k++)
      {
        // only calculate upper triangular part
        for(int j = 0; j < r2; j++)
          for(int i = 0; i <= j; i++)
          {
            Chunk<T> tmp1{};
            for(int jb_ = 0; jb_ < r1; jb_+=2)
            {
              // 2-way unrolling (unrollSize == 2)
              Chunk<T> tmp20{};
              Chunk<T> tmp21{};
              for(int i_ = 0; i_ < r1; i_++)
              {
                fmadd(last_t2(i_,jb_+0),subT.chunk(i_,k,i),tmp20);
                fmadd(last_t2(i_,jb_+1),subT.chunk(i_,k,i),tmp21);
              }
              fmadd(subT.chunk(jb_,k,j),tmp20,tmp1);
              if( jb_+1 < r1 )
                fmadd(subT.chunk(jb_+1,k,j),tmp21,tmp1);
            }
            // this directly works on a pointer for the data of t2 to allow an OpenMP array reduction
            t2data[i+j*r2] += sum(tmp1);
          }
      }
      // copy upper triangular part to lower triangular part (symmetric!)
      for(int j = 0; j < r2; j++)
        for(int i = j+1; i < r2; i++)
          t2(i,j) = t2(j,i);
    }
    //wtime = omp_get_wtime()-wtime;
    //std::cout << "GFlop/s: " << flops/wtime*1.e-9 << std::endl;
    return std::sqrt(t2(0,0));
  }

}


#endif // PITTS_TENSORTRAIN_NORM_HPP
