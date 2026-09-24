// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_from_dense_twosided_impl.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm, two-sided variant)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-08
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP

// includes
#include <numeric>
#include <iostream>
#include "pitts_tensortrain_from_dense_twosided.hpp"
#include "pitts_eigen.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper namespace for two-sided TT-SVD functionality
    namespace TTSVD_twoSided
    {
      //! small wrapper around SVD only with data size
      template<typename T, class TensorType>
      auto split_svd_only(const TensorType& M)
      {
        auto n = ConstEigenMap(M).rows();
        const auto m = ConstEigenMap(M).cols();

        constexpr auto pow3 = [](double x){return x*x*x;};

        const auto timer = PITTS::performance::createScopedTimer<TensorType>(
          {{"n", "m"},{n, m}}, // arguments
          {{(4.*n*m*std::min(n,m)+2*pow3(std::min(n,m))/3)*kernel_info::FMA<T>()}, // flops
           {(n*m)*kernel_info::Load<T>() + ((n+m+1)*std::min(n,m))*kernel_info::Store<T>()}} // data transfers
          );

        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        const auto mapM = ConstEigenMap(M);

#if EIGEN_VERSION_AT_LEAST(3,4,90)
        auto svd = Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU>(mapM);
#else
        auto svd = Eigen::BDCSVD<EigenMatrix>(mapM, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

        assert(!std::isnan(svd.singularValues()(0)));

        return svd;
      }

      template<typename T>
      void split(const MultiVector<T>& X, MultiVector<T>& Y, Tensor2<T>& M, int nextDim, T rankTolerance, int maxRank, T& initialFrobeniusNorm)
      {
        // timer
        const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

        std::cout << "TT-SVD::split  matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
        auto svd = [&](){
          if( X.rows() > 10*X.cols() )
          {
            // calculate QR decomposition (QR-trick: X=QR,SVD(R))
            block_TSQR(X, M, 0, false);
            return split_svd_only<T>(M);
          }
          else
          {
            return split_svd_only<T>(X);
          }
        }();

        if( initialFrobeniusNorm == 0 )
          initialFrobeniusNorm = svd.singularValues().norm();

        int rank = internal::rankInFrobeniusNorm(svd, rankTolerance*initialFrobeniusNorm);
        std::cout << "n: " << svd.singularValues().size() << ", rank: " << rank << ", maxRank: " << maxRank << ", rankTol*initialFrobeniusNorm: " << rankTolerance * initialFrobeniusNorm << "\n";
        if( maxRank >= 0 )
          rank = std::min(rank, maxRank);

        //std::cout << "singular values: " << svd.singularValues().transpose()/initialFrobeniusNorm << "\n";

        // copy right singular vectors
        M.resize(X.cols(), rank);
        EigenMap(M).noalias() = svd.matrixV().leftCols(rank);

        // transform input, s.t. Y \approx X M
        transform(X, M, Y, {X.rows()/nextDim, rank*nextDim});
      }
    }
  }


  // implement TT fromDense twosided
  template<typename T>
  TensorTrain<T> fromDense_twoSided(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance, int maxRank)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // abort early for zero dimensions
    if( dimensions.size() == 0 )
    {
      if( X.rows()*X.cols() !=  0 )
        throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");
      return TensorTrain<T>{dimensions};
    }

    const auto totalSize = std::accumulate(begin(dimensions), end(dimensions), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    const auto nDim = dimensions.size();
    if( X.rows() != totalSize/dimensions[nDim-1] || X.cols() != dimensions[nDim-1] )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    // actually convert to tensor train format
    std::vector<Tensor3<T>> subTensors(nDim);
    Tensor2<T> M;
    using RealType = decltype(std::abs(T(1)));
    const RealType rankTol = std::abs(rankTolerance) / std::sqrt(RealType(nDim-1));
    RealType initialFrobeniusNorm = 0;
    for(int ii = 0; ii < nDim; ii++)
    {
      if( ii % 2 == 0 )
      {
        // right part
        const auto iDim = nDim - 1 - ii/2;
        if( ii > 0 )
        {
          const auto r1 = subTensors[iDim+1].r1();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.rows()*work.cols())/(n*r1), n*r1}, true);
        }
        if( ii != nDim-1 )
        {
          internal::TTSVD_twoSided::split(X, work, M, 1, rankTol, maxRank, initialFrobeniusNorm);
          fold_right(ConstEigenMap(M).transpose(), dimensions[iDim], subTensors[iDim]);
        }
        else
        {
          fold_right(X, dimensions[iDim], subTensors[iDim]);
        }

      }
      else
      {
        // left part
        const auto iDim = ii / 2;
        {
          const auto r2 = (iDim == 0 ) ? 1 : subTensors[iDim-1].r2();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.cols()*work.rows())/(n*r2), n*r2}, false);
        }
        if( ii != nDim - 1 )
        {
          internal::TTSVD_twoSided::split(X, work, M, 1, rankTol, maxRank, initialFrobeniusNorm);
          fold_left(M, dimensions[iDim], subTensors[iDim]);
        }
        else
        {
          fold_left(ConstEigenMap(X).transpose(), dimensions[iDim], subTensors[iDim]);
        }
      }
    }

    TensorTrain<T> result(dimensions);
    result.setSubTensors(0, std::move(subTensors));

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP
