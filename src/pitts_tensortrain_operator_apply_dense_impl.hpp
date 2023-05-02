/*! @file pitts_tensortrain_operator_apply_dense_impl.hpp
* @brief apply a tensor train operator to a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_IMPL_HPP

// includes
#include <cmath>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_multivector.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and (flattened) dense tensor
    template<typename T>
    void apply_dense_contract(const TensorTrainOperator<T>& TTOp, int iDim, const Tensor3<T>& Aop, const MultiVector<T>& x, MultiVector<T>& y)
    {
      constexpr auto partial_products = [](const auto& v, int middle)
      {
        assert(middle < v.size());
        const long long n_left = std::accumulate(v.begin(), v.begin()+middle, 1, std::multiplies<long long>());
        const long long n_right = std::accumulate(v.begin()+middle+1, v.end(), 1, std::multiplies<long long>());
        return std::make_pair(n_left, n_right);
      };
      
      const long long nCols = x.cols();
      const long long n = TTOp.row_dimensions()[iDim];
      const long long m = TTOp.column_dimensions()[iDim];
      const long long r1 = Aop.r1();
      const long long r2 = Aop.r2();
      const int nDim = TTOp.row_dimensions().size();
      const auto [n_left, n_right] = partial_products(TTOp.row_dimensions(), iDim);
      const auto [m_left, m_right] = partial_products(TTOp.column_dimensions(), iDim);

      // copy operator to better format
      // todo simpler memory layout for Tensor3 -> this is not needed any more...
      Eigen::MatrixX<T> tmpA(r1*n,m*r2);
#pragma omp parallel for collapse(3) schedule(static)
      for(long long k2 = 0; k2 < r2; k2++)
        for(long long j = 0; j < m; j++)
          for(long long i = 0; i < n; i++)
            for(long long k1 = 0; k1 < r1; k1++)
              tmpA(k1+i*r1, j+m*k2) = Aop(k1, TTOp.index(iDim, i, j), k2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"nCols", "m_left", "r2", "n", "n_right", "m", "r1"},{nCols, m_left, r2, n, n_right, m, r1}}, // arguments
        {{nCols*m_right*r2*n*n_left*m*r1*kernel_info::FMA<T>()}, // flops
         {(r1*n*m*r2+nCols*n_left*r1*m*m_right)*kernel_info::Load<T>() + (nCols*n_left*n*r2*m_right)*kernel_info::Store<T>()}} // data transfers
        );

      // x should have m_left * m * r2 * n_right #rows
      // y should get m_left * r1 * n * n_right #rows
      // both stored column major in this order...
      assert(m_left * m * r2 * n_right == x.rows());
      y.resize(m_left * r1 * n * n_right, nCols);

      const auto r1n = r1*n;
      const auto mr2 = m*r2;

      for(int iCol = 0; iCol < nCols; iCol++)
      {
        if( m_left == 1 )
        {
            Eigen::Map<Eigen::MatrixX<T>> yMap(&y(0, iCol), r1n, n_right);
            Eigen::Map<const Eigen::MatrixX<T>> xMap(&x(0, iCol), mr2, n_right);
            yMap.noalias() = tmpA * xMap;
        }
        else if( n_right == 1 )
        {
          Eigen::Map<Eigen::MatrixX<T>> yMap(&y(0, iCol), m_left, r1n);
          Eigen::Map<const Eigen::MatrixX<T>> xMap(&x(0, iCol), m_left, mr2);
          yMap.noalias() = xMap * tmpA.transpose();
        }
        else // n_right and m_left != 1
        {
#ifndef __clang__
#pragma omp parallel for schedule(dynamic) if(n_right > 50)
#endif
          for(int ir = 0; ir < n_right; ir++)
          {
            Eigen::Map<Eigen::MatrixX<T>> yMap(&y(ir * m_left * r1n, iCol), m_left, r1n);
            Eigen::Map<const Eigen::MatrixX<T>> xMap(&x(ir * m_left * mr2, iCol), m_left, mr2);
            yMap.noalias() = xMap * tmpA.transpose();
          }
        }
      }
    }
  }

  // implement TT Op apply to dense
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOp, const MultiVector<T>& MVx, MultiVector<T>& MVy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for matching dimensions
    const long mTotal = std::accumulate(TTOp.column_dimensions().begin(), TTOp.column_dimensions().end(), 1, std::multiplies<long>());
    if( mTotal != MVx.rows() )
      throw std::invalid_argument("TensorTrainOperator: input tensor dimension mismatch!");

    MultiVector<T> mv_tmp;

    // perform actual calculation
    const int nDim = TTOp.row_dimensions().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensor(iDim);
      const auto& x = (iDim == nDim-1) ? MVx : MVy;
      auto& y = mv_tmp;

      internal::apply_dense_contract(TTOp, iDim, subTOp, x, y);

      std::swap(y, MVy);
    }
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_IMPL_HPP
