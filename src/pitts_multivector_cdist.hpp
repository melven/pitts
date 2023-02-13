/*! @file pitts_multivector_cdist.hpp
* @brief calculate the distance of each vector in one multi-vector with each vector in another multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_multivector_cdist.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_MULTIVECTOR_CDIST)
import pitts_multivector_cdist;
#define PITTS_MULTIVECTOR_CDIST_HPP
#endif

// include guard
#ifndef PITTS_MULTIVECTOR_CDIST_HPP
#define PITTS_MULTIVECTOR_CDIST_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <memory>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_multivector_cdist;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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

    // gather performance data
    const double rowsd = X.rows();
    const double nd = n, md = m;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "Xcols", "Ycols"},{X.rows(),n,m}}, // arguments
        {{(nd*md*rowsd + nd*rowsd + md*rowsd)*kernel_info::FMA<T>()}, // flops
         {(nd*rowsd + md*rowsd)*kernel_info::Load<T>() + (nd*md)*kernel_info::Store<T>()}} // data transfers
        );

    D.resize(n,m);

    constexpr long long blockSize = 10; // required for reducing memory traffic of Y (X is loaded only once, Y n/blockSize times)

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> tmpX(new Chunk<T>[blockSize]);
      std::unique_ptr<Chunk<T>[]> tmpXY(new Chunk<T>[m*blockSize]);
#pragma omp for schedule(static)
      for(long long iB = 0; iB < n; iB+=blockSize)
      {
        const auto nb = std::min(blockSize, n-iB);
        for(long long i = 0; i < nb; i++)
          tmpX[i] = Chunk<T>{};

        for(long long i = 0; i < nb; i++)
          for(long long j = 0; j < m; j++)
            tmpXY[i+j*blockSize] = Chunk<T>{};

        for(long long c = 0; c < chunks; c++)
        {
          for(long long i = 0; i < nb; i++)
          {
            fmadd(X.chunk(c,iB+i), X.chunk(c,iB+i), tmpX[i]);
            for(long long j = 0; j < m; j++)
              fmadd(X.chunk(c,iB+i), Y.chunk(c,j), tmpXY[i+j*blockSize]);
          }
        }

        for(long long j = 0; j < m; j++)
          for(long long i = 0; i < nb; i++)
            D(iB+i,j) = sum(tmpX[i]) - 2 * sum(tmpXY[i+j*blockSize]);
      }
    }

#pragma omp parallel for schedule(static)
    for(long long j = 0; j < m; j++)
    {
      Chunk<T> tmpY = Chunk<T>{};

      for(long long c = 0; c < chunks; c++)
        fmadd(Y.chunk(c,j), Y.chunk(c,j), tmpY);

      const auto s = sum(tmpY);
      for(long long i = 0; i < n; i++)
        D(i,j) += s;
    }

  }

  // explicit template instantiations
  template void cdist2<float>(const MultiVector<float>& X, const MultiVector<float>& Y, Tensor2<float>& D);
  template void cdist2<double>(const MultiVector<double>& X, const MultiVector<double>& Y, Tensor2<double>& D);
  // workaround for pybind interface (no matching function call) with C++20 modules
  template std::string_view PITTS::internal::TypeName::name<float>();
  template std::string_view PITTS::internal::TypeName::name<double>();
}


#endif // PITTS_MULTIVECTOR_CDIST_HPP
