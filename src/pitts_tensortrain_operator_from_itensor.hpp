/*! @file pitts_tensortrain_operator_from_itensor.hpp
* @brief conversion of a ITensor MPO to PITTS::TensorTrainOperator (just copying the data)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-16
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_FROM_ITENSOR_HPP
#define PITTS_TENSORTRAIN_OPERATOR_FROM_ITENSOR_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_timer.hpp"
#include "itensor/mps/mpo.h"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! copy a ITensor MPO to PITTS::TensorTrainOperator
  //!
  //! @warning Correct data type must be specified, MPO can be both real or complex (without beeing templetized?)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param mpo   input tensor train operator in ITensor MPO format
  //! @return      resulting tensor train operator
  //!
  template<typename T>
  TensorTrainOperator<T> fromITensor(const itensor::MPO& mpo)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // get ITensor indices of row dimensions
    const auto [row_idx, col_idx] = [&mpo]()
    {
      std::vector<itensor::Index> idx1(length(mpo));
      std::vector<itensor::Index> idx2(length(mpo));
      for(int iDim = 0; iDim < idx1.size(); iDim++)
      {
        const auto tmp  = siteInds(mpo, iDim+1);
        idx1[iDim] = tmp(1);
        idx2[iDim] = tmp(2);
      }
      return std::make_tuple(
          itensor::IndexSet(idx1),
          itensor::IndexSet(idx2));
    }();

    // get dimensions as integers from Index arrays
    constexpr auto extractDims = [](const itensor::IndexSet& idx)
    {
      std::vector<int> dims(idx.size());
      for(int iDim = 0; iDim < idx.size(); iDim++)
        dims[iDim] = dim(idx[iDim]);
      return dims;
    };
    const auto row_dims = extractDims(row_idx);
    const auto col_dims = extractDims(col_idx);


    TensorTrainOperator<T> ttOp(row_dims, col_dims);


    // get ranks from the itensor mpo
    const auto link_idx = [&mpo]()
    {
      std::vector<itensor::Index> idx(length(mpo)-1);
      for(int iLink = 0; iLink < idx.size(); iLink++)
        idx[iLink] = linkIndex(mpo, iLink+1);
      return itensor::IndexSet(idx);
    }();

    const auto ranks = extractDims(link_idx);
    ttOp.setTTranks(ranks);


    for(int iDim = 0; iDim < row_dims.size(); iDim++)
    {
      auto& subT = ttOp.tensorTrain().editableSubTensors()[iDim];
      
      if( iDim == 0 )
      {
          for(int j = 0; j < row_dims[iDim]; j++)
            for(int k = 0; k < col_dims[iDim]; k++)
              for(int l = 0; l < ranks[iDim]; l++)
                subT(0, ttOp.index(iDim, j, k), l) = elt(mpo(iDim+1), row_idx[iDim]=j+1, col_idx[iDim]=k+1, link_idx[iDim]=l+1);
      }
      else if( iDim+1 < row_dims.size() )
      {
        for(int i = 0; i < ranks[iDim-1]; i++)
          for(int j = 0; j < row_dims[iDim]; j++)
            for(int k = 0; k < col_dims[iDim]; k++)
              for(int l = 0; l < ranks[iDim]; l++)
                subT(i, ttOp.index(iDim, j, k), l) = elt(mpo(iDim+1), link_idx[iDim-1]=i+1, row_idx[iDim]=j+1, col_idx[iDim]=k+1, link_idx[iDim]=l+1);
      }
      else
      {
        for(int i = 0; i < ranks[iDim-1]; i++)
          for(int j = 0; j < row_dims[iDim]; j++)
            for(int k = 0; k < col_dims[iDim]; k++)
                subT(i, ttOp.index(iDim, j, k), 0) = elt(mpo(iDim+1), link_idx[iDim-1]=i+1, row_idx[iDim]=j+1, col_idx[iDim]=k+1);
      }
    }

    return ttOp;
  }

}

#endif // PITTS_TENSORTRAIN_OPERATOR_FROM_ITENSOR_HPP
