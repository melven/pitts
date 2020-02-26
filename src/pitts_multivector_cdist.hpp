/*! @file pitts_multivector_cdist.hpp
* @brief calculate the distance of each vector in one multi-vector with each vector in another multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_CDIST_HPP
#define PITTS_MULTIVECTOR_CDIST_HPP

// includes
#include <vector>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the squared distance of each vector in one multi-vector with each vector in another multi-vector
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @param X  first multi-vector  (assumed to have a high number of columns)
  //! @param Y  second multi-vector (assumed to have a low number of columns)
  //! @param D  pair-wise squared distance of each column in X and Y, dimension (X.cols() x Y.rows())
  //!
  template<typename T>
  void cdist2(const MultiVector<T>& X, const MultiVector<T>& Y, Tensor2<T>& D)
  {
    // exploit <x-y,x-y> = ||x||^2 - 2<x,y> + ||y||^2
    const auto chunks = X.rowChunks();
    const auto n = X.cols();
    const auto m = Y.cols();
    D.resize(n,m);

    constexpr int blockSize = 10; // required for reducing memory traffic of Y (X is loaded only once, Y n/blockSize times)

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> tmpX(new Chunk<T>[blockSize]);
      std::unique_ptr<Chunk<T>[]> tmpXY(new Chunk<T>[m*blockSize]);
#pragma omp for schedule(static)
      for(int iB = 0; iB < n; iB+=blockSize)
      {
        const auto nb = std::min(blockSize, n-iB);
        for(int i = 0; i < nb; i++)
          tmpX[i] = Chunk<T>{};

        for(int i = 0; i < nb; i++)
          for(int j = 0; j < m; j++)
            tmpXY[i+j*blockSize] = Chunk<T>{};

        for(int c = 0; c < chunks; c++)
        {
          for(int i = 0; i < nb; i++)
          {
            fmadd(X.chunk(c,iB+i), X.chunk(c,iB+i), tmpX[i]);
            for(int j = 0; j < m; j++)
              fmadd(X.chunk(c,iB+i), Y.chunk(c,j), tmpXY[i+j*blockSize]);
          }
        }

        for(int j = 0; j < m; j++)
          for(int i = 0; i < nb; i++)
            D(iB+i,j) = sum(tmpX[i]) - 2 * sum(tmpXY[i+j*blockSize]);
      }
    }

#pragma omp parallel for schedule(static)
    for(int j = 0; j < m; j++)
    {
      Chunk<T> tmpY = Chunk<T>{};

      for(int c = 0; c < chunks; c++)
        fmadd(Y.chunk(c,j), Y.chunk(c,j), tmpY);

      const auto s = sum(tmpY);
      for(int i = 0; i < n; i++)
        D(i,j) += s;
    }

  }

}


#endif // PITTS_MULTIVECTOR_CDIST_HPP
