/*! @file pitts_tensortrain_operator_to_qtt_impl.hpp
* @brief conversion from the tensor-train operator format into a dense matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-06-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_TO_QTT_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_TO_QTT_IMPL_HPP

// includes
#include "pitts_tensortrain_operator_to_qtt.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TTOp toQtt
  template<typename T>
  TensorTrainOperator<T> toQtt(const TensorTrainOperator<T>& TTOp)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    const int nDim = TTOp.row_dimensions().size();

    std::vector<int> newRowDims;
    std::vector<int> newColDims;
    std::vector<Tensor3<T>> newSubTensors;

    Tensor3<T> subT, tmpSubT;
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      copy(TTOp.tensorTrain().subTensor(iDim), subT);
      long long n = TTOp.row_dimensions()[iDim];
      long long m = TTOp.column_dimensions()[iDim];
      while(n % 2 == 0 && m % 2 == 0 && n > 2 && m > 2)
      {
        // we need to reorder the entries!
        const long long r1 = subT.r1();
        const long long r2 = subT.r2();
        tmpSubT.resize(r1, n*m, r2);
        for(long long k2 = 0; k2 < r2; k2++)
          for(long long j = 0; j < m/2; j++)
            for(long long i = 0; i < n/2; i++)
              for(long long k1 = 0; k1 < r1; k1++)
              {
                tmpSubT(k1, 0 + 0*2 + i*2*2 + j*2*n,k2) = subT(k1, (0+i*2) + (0+j*2)*n, k2);
                tmpSubT(k1, 1 + 0*2 + i*2*2 + j*2*n,k2) = subT(k1, (1+i*2) + (0+j*2)*n, k2);
                tmpSubT(k1, 0 + 1*2 + i*2*2 + j*2*n,k2) = subT(k1, (0+i*2) + (1+j*2)*n, k2);
                tmpSubT(k1, 1 + 1*2 + i*2*2 + j*2*n,k2) = subT(k1, (1+i*2) + (1+j*2)*n, k2);
              }
        
        auto [t3a, t3b] = split(tmpSubT, 2*2, n/2*m/2, false);
        newRowDims.push_back(2);
        newColDims.push_back(2);
        newSubTensors.push_back(std::move(t3a));

        std::swap(subT, t3b);
        n /= 2;
        m /= 2;
      }
      newRowDims.push_back(n);
      newColDims.push_back(m);
      newSubTensors.push_back(std::move(subT));
      subT = Tensor3<T>(0,0,0);
    }

    TensorTrainOperator<T> qttOp(newRowDims, newColDims);
    qttOp.tensorTrain().setSubTensors(0, std::move(newSubTensors));

    return qttOp;
  }
}


#endif // PITTS_TENSORTRAIN_OPERATOR_TO_QTT_IMPL_HPP
