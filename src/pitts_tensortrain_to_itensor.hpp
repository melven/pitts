/*! @file pitts_tensortrain_to_itensor.hpp
* @brief conversion of a PITTS::TensorTrain to ITensor MPS (just copying the data)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-16
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_TO_ITENSOR_HPP
#define PITTS_TENSORTRAIN_TO_ITENSOR_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "itensor/mps/mps.h"
#include "itensor/itensor.h"
#include <vector>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! copy a PITTS::TensorTrain to an ITensor MPS
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param tt    input tensor train in pitts::TensorTrain format
  //! @return      resulting itensor mps
  //!
  template<typename T>
  itensor::MPS toITensor(const TensorTrain<T>& tt)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // set up dimensions
    const auto sites = [&tt]()
    {
      const int N = tt.dimensions().size();
      auto sites = itensor::SiteSet(N);
      for(int iDim = 0; iDim < N; iDim++)
      {
        const auto d = tt.dimensions()[iDim];
        const auto idx = itensor::Index(d, itensor::format("Site,n=%d", iDim+1));
        sites.set(iDim+1, itensor::GenericSite(idx));
      }
      return sites;
    }();

    // inspired by the  new_tensors function in mps.cc
    // (I didn't find a nice way to set individual bond dimensions, so we will just construct all ITensors ourselves)
    // set up ranks
    const auto links = [&tt]()
    {
      const auto ranks = tt.getTTranks();
      std::vector<itensor::Index> links(ranks.size());
      for(int i = 0; i < ranks.size(); i++)
        links[i] = itensor::Index(ranks[i], itensor::format("Link,l=%d",i+1));
      return itensor::IndexSet{links};
    }();


    auto mps = itensor::MPS(tt.dimensions().size());


    for(int iDim = 0; iDim < tt.dimensions().size(); iDim++)
    {
      const auto& subT = tt.subTensor(iDim);
      
      if( iDim == 0 )
      {
        const auto idx_j = sites(iDim+1);
        const auto idx_k = links(iDim+1);

        auto mps_nA = itensor::ITensor(idx_j, idx_k);

        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            mps_nA.set(idx_j=j+1, idx_k=k+1, subT(0, j, k));

        mps.set(iDim+1, std::move(mps_nA));
      }
      else if( iDim+1 < tt.dimensions().size() )
      {
        const auto idx_i = itensor::dag(links(iDim));
        const auto idx_j = sites(iDim+1);
        const auto idx_k = links(iDim+1);

        auto mps_nA = itensor::ITensor(idx_i, idx_j, idx_k);

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < subT.n(); j++)
            for(int k = 0; k < subT.r2(); k++)
              mps_nA.set(idx_i=i+1, idx_j=j+1, idx_k=k+1, subT(i, j, k));

        mps.set(iDim+1, std::move(mps_nA));
      }
      else
      {
        const auto idx_i = itensor::dag(links(iDim));
        const auto idx_j = sites(iDim+1);

        auto mps_nA = itensor::ITensor(idx_i, idx_j);

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < subT.n(); j++)
            mps_nA.set(idx_i=i+1, idx_j=j+1, subT(i, j, 0));

        mps.set(iDim+1, std::move(mps_nA));
      }
    }

    return mps;
  }

}

#endif // PITTS_TENSORTRAIN_TO_ITENSOR_HPP
