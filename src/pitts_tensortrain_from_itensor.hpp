/*! @file pitts_tensortrain_from_itensor.hpp
* @brief conversion of a ITensor MPS to PITTS::TensorTrain (just copying the data)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-14
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_ITENSOR_HPP
#define PITTS_TENSORTRAIN_FROM_ITENSOR_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "itensor/mps/mps.h"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! copy a ITensor MPS to PITTS::TensorTrain
  //!
  //! @warning Correct data type must be specified, MPS can be both real or complex (without beeing templetized?)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param mps   input tensor train in ITensor MPS format
  //! @return      resulting tensor train
  //!
  template<typename T>
  TensorTrain<T> fromITensor(const itensor::MPS& mps)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // get dimensions from the itensor mps
    const auto dims = [&mps]()
    {
      std::vector<int> dims(length(mps));
      for(int iDim = 0; iDim < dims.size(); iDim++)
        dims[iDim] = dim(siteIndex(mps, iDim+1));
      return dims;
    }();

    std::vector<Tensor3<T>> subTensors(dims.size());
    for(int iDim = 0; iDim < dims.size(); iDim++)
    {
      auto& subT = subTensors[iDim];
      
      if( iDim == 0 )
      {
        const auto idx_j = siteIndex(mps, iDim+1);
        const auto idx_k = rightLinkIndex(mps, iDim+1);
        subT.resize(1, dim(idx_j), dim(idx_k));

        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            subT(0, j, k) = elt(mps(iDim+1), idx_j=j+1, idx_k=k+1);
      }
      else if( iDim+1 < dims.size() )
      {
        const auto idx_i = leftLinkIndex(mps, iDim+1);
        const auto idx_j = siteIndex(mps, iDim+1);
        const auto idx_k = rightLinkIndex(mps, iDim+1);
        subT.resize(dim(idx_i), dim(idx_j), dim(idx_k));

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < subT.n(); j++)
            for(int k = 0; k < subT.r2(); k++)
              subT(i, j, k) = elt(mps(iDim+1), idx_i=i+1, idx_j=j+1, idx_k=k+1);
      }
      else
      {
        const auto idx_i = leftLinkIndex(mps, iDim+1);
        const auto idx_j = siteIndex(mps, iDim+1);
        subT.resize(dim(idx_i), dim(idx_j), 1);

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < subT.n(); j++)
            subT(i, j, 0) = elt(mps(iDim+1), idx_i=i+1, idx_j=j+1);
      }
    }

    TensorTrain<T> tt(dims);
    tt.setSubTensors(0, std::move(subTensors));

    return tt;
  }

}

#endif // PITTS_TENSORTRAIN_FROM_ITENSOR_HPP
