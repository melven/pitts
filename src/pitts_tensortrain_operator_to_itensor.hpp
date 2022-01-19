/*! @file pitts_tensortrain_operator_to_itensor.hpp
* @brief conversion of a PITTS::TensorTrainOperator to ITensor MPO (just copying the data)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-17
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_TO_ITENSOR_HPP
#define PITTS_TENSORTRAIN_OPERATOR_TO_ITENSOR_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_timer.hpp"
#include "itensor/mps/mpo.h"
#include "itensor/itensor.h"
#include <vector>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! copy a PITTS::TensorTrainOperator to an ITensor MPO
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param tt    input tensor train operator in pitts::TensorTrainOperator format
  //! @return      resulting itensor mpo
  //!
  template<typename T>
  itensor::MPO toITensor(const TensorTrainOperator<T>& ttOp)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // set up dimensions
    const auto row_sites = [&ttOp]()
    {
      const int N = ttOp.row_dimensions().size();
      auto sites = itensor::SiteSet(N);
      for(int iDim = 0; iDim < N; iDim++)
      {
        const auto d = ttOp.row_dimensions()[iDim];
        const auto idx = itensor::Index(d, itensor::format("Site,n=%d", iDim+1));
        sites.set(iDim+1, itensor::GenericSite(idx));
      }
      return sites;
    }();

    const auto col_sites = [&ttOp,&row_sites]()
    {
      const int N = ttOp.column_dimensions().size();
      auto sites = itensor::SiteSet(N);
      for(int iDim = 0; iDim < N; iDim++)
      {
        if( ttOp.row_dimensions()[iDim] == ttOp.column_dimensions()[iDim] )
          sites.set(iDim+1, itensor::GenericSite(prime(row_sites(iDim+1))));
        else
        {
          const auto d = ttOp.column_dimensions()[iDim];
          const auto idx = itensor::Index(d, itensor::format("Site,n=%d", iDim+1));
          sites.set(iDim+1, itensor::GenericSite(idx));
        }
      }
      return sites;
    }();

    // inspired by the  new_tensors function in mps.cc
    // (I didn't find a nice way to set individual bond dimensions, so we will just construct all ITensors ourselves)
    // set up ranks
    const auto links = [&ttOp]()
    {
      const auto ranks = ttOp.getTTranks();
      std::vector<itensor::Index> links(ranks.size());
      for(int i = 0; i < ranks.size(); i++)
        links[i] = itensor::Index(ranks[i], itensor::format("Link,l=%d",i+1));
      return itensor::IndexSet{links};
    }();


    auto mpo = itensor::MPO(ttOp.row_dimensions().size());


    for(int iDim = 0; iDim < ttOp.row_dimensions().size(); iDim++)
    {
      const auto& subT = ttOp.tensorTrain().subTensors()[iDim];
      
      if( iDim == 0 )
      {
        const auto idx_j = row_sites(iDim+1);
        const auto idx_k = dag(col_sites(iDim+1));
        const auto idx_l = links(iDim+1);

        auto mpo_nA = itensor::ITensor(idx_j, idx_k, idx_l);

        for(int j = 0; j < ttOp.row_dimensions()[iDim]; j++)
          for(int k = 0; k < ttOp.column_dimensions()[iDim]; k++)
            for(int l = 0; l < subT.r2(); l++)
              mpo_nA.set(idx_j=j+1, idx_k=k+1, idx_l=l+1, subT(0, ttOp.index(iDim, j, k), l));

        mpo.set(iDim+1, std::move(mpo_nA));
      }
      else if( iDim+1 < ttOp.row_dimensions().size() )
      {
        const auto idx_i = dag(links(iDim));
        const auto idx_j = row_sites(iDim+1);
        const auto idx_k = dag(col_sites(iDim+1));
        const auto idx_l = links(iDim+1);

        auto mpo_nA = itensor::ITensor(idx_i, idx_j, idx_k, idx_l);

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < ttOp.row_dimensions()[iDim]; j++)
            for(int k = 0; k < ttOp.column_dimensions()[iDim]; k++)
              for(int l = 0; l < subT.r2(); l++)
                mpo_nA.set(idx_i=i+1, idx_j=j+1, idx_k=k+1, idx_l=l+1, subT(i, ttOp.index(iDim, j, k), l));

        mpo.set(iDim+1, std::move(mpo_nA));
      }
      else
      {
        const auto idx_i = dag(links(iDim));
        const auto idx_j = row_sites(iDim+1);
        const auto idx_k = dag(col_sites(iDim+1));

        auto mpo_nA = itensor::ITensor(idx_i, idx_j, idx_k);

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < ttOp.row_dimensions()[iDim]; j++)
            for(int k = 0; k < ttOp.column_dimensions()[iDim]; k++)
                mpo_nA.set(idx_i=i+1, idx_j=j+1, idx_k=k+1, subT(i, ttOp.index(iDim, j, k), 0));

        mpo.set(iDim+1, std::move(mpo_nA));
      }
    }

    return mpo;
  }

}

#endif // PITTS_TENSORTRAIN_OPERATOR_TO_ITENSOR_HPP
