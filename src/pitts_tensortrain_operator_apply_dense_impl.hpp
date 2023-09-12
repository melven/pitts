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
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector_reshape.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_eigen.hpp"
#include "pitts_parallel.hpp"

#define PITTS_DIRECT_GEMM


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const double alpha, const double * A, const CBLAS_INDEX lda, const double * B, const CBLAS_INDEX ldb, const double beta, double * C, const CBLAS_INDEX ldc)
    {
      cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const float alpha, const float * A, const CBLAS_INDEX lda, const float * B, const CBLAS_INDEX ldb, const float beta, float * C, const CBLAS_INDEX ldc)
    {
      cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }


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

      Eigen::Map<const Eigen::MatrixX<T>> tmpA(&Aop(0,0,0), r1n, mr2);

      for(int iCol = 0; iCol < nCols; iCol++)
      {
        if( m_left == 1 )
        {
          Eigen::Map<Eigen::MatrixX<T>> yMap(&y(0, iCol), r1n, n_right);
          Eigen::Map<const Eigen::MatrixX<T>> xMap(&x(0, iCol), mr2, n_right);
#ifndef PITTS_DIRECT_GEMM
          yMap.noalias() = tmpA * xMap;
#else
          cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasNoTrans, tmpA.rows(), xMap.cols(), tmpA.cols(), T(1), tmpA.data(), tmpA.colStride(), xMap.data(), xMap.colStride(), T(0), yMap.data(), yMap.colStride());
#endif
        }
        else if( n_right == 1 )
        {
          Eigen::Map<Eigen::MatrixX<T>> yMap(&y(0, iCol), m_left, r1n);
          Eigen::Map<const Eigen::MatrixX<T>> xMap(&x(0, iCol), m_left, mr2);
#ifndef PITTS_DIRECT_GEMM
          yMap.noalias() = xMap * tmpA.transpose();
#else
          cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasTrans, xMap.rows(), tmpA.rows(), xMap.cols(), T(1), xMap.data(), xMap.colStride(), tmpA.data(), tmpA.colStride(), T(0), yMap.data(), yMap.colStride());
#endif
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
#ifndef PITTS_DIRECT_GEMM
            yMap.noalias() = xMap * tmpA.transpose();
#else
            cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasTrans, xMap.rows(), tmpA.rows(), xMap.cols(), T(1), xMap.data(), xMap.colStride(), tmpA.data(), tmpA.colStride(), T(0), yMap.data(), yMap.colStride());
#endif
          }
        }
      }
    }

    //! assumes a very special memory layout: A(:,(*,1:rA)) * x((:,:),*)^T -> y((:,:),(:,1:rA))
    template<typename T>
    void apply_dense_contract_padded(long long r, long long rA, const MultiVector<T>& x, const MultiVector<T>& A, MultiVector<T>& y, std::vector<MultiVector<T>>& tmpx, std::vector<MultiVector<T>>& tmpy)
    {
      const auto yn = A.rows();
      const auto xn = x.rows() / r;
      const auto m = A.cols() / rA;
      if( A.cols() % rA != 0 || x.cols() != m || x.rows() % r != 0 )
        throw std::invalid_argument("apply_dense_contract: invalid dimensions!");

      y.resize(yn*xn, r*rA, false);
      
      const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
          {{"yn", "xn", "m", "rA", "r"},{yn, xn, m, rA, r}}, // arguments
          {{double(xn*r*m*yn*rA)*kernel_info::FMA<T>()}, // flops
          {double(xn*r*m + yn*m*rA)*kernel_info::Load<T>() + double(yn*xn*r*rA)*kernel_info::Store<T>()}} // data transfers
          );
      
      using mat = Eigen::MatrixX<T>;

      // calculate the compute intensity of the optimal GEMM and loop over GEMMS
      const double Ic = (xn*r*m*yn*rA)*2. / (8.*(xn*r*m + yn*m*rA + yn*xn*r*rA));
      const double Ic_loop = (xn*m*yn)*2. / (8.*(xn*m + yn*m + yn*xn));
      //std::cout << "Ic: " << Ic << ", Ic_loop: " << Ic_loop << "\n";
      if( true ) //if( Ic > 20 )
      {
#pragma omp parallel
        {
          const auto [iThread, nThreads] = parallel::ompThreadInfo();
          const auto [rfirst, rlast] = parallel::distribute(r, {iThread, nThreads});
          const auto rlocal = rlast - rfirst + 1;

          for(long long iA = 0; iA < rA; iA++)
          {
            const auto xstride = Eigen::OuterStride<>(&x(0,1) - &x(0,0));
            const auto Astride = Eigen::OuterStride<>(&A(0,1) - &A(0,0));

            using map = Eigen::Map<mat, Eigen::Unaligned, Eigen::OuterStride<>>;
            using const_map = Eigen::Map<const mat, Eigen::Unaligned, Eigen::OuterStride<>>;

            const_map mapx(&x(rfirst*xn,0), xn*rlocal, m, xstride);
            const_map mapA(&A(0,m*iA), yn, m, Astride);

            MultiVector<T>& tmpy_ = tmpy[iThread];
            tmpy_.resize(yn, xn*rlocal);
            auto mapy = EigenMap(tmpy_);

#ifndef PITTS_DIRECT_GEMM
            mapy.noalias() = mapA * mapx.transpose();
#else
            cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasTrans, mapA.rows(), mapx.rows(), mapA.cols(), T(1), mapA.data(), mapA.colStride(), mapx.data(), mapx.colStride(), T(0), mapy.data(), mapy.colStride());
#endif
            for(long long i = rfirst; i <= rlast; i++)
              for(long long ixn = 0; ixn < xn; ixn++)
              {
                //for(long long iyn = 0; iyn < yn; iyn++)
                //  y(iyn+yn*ixn, i+iA*r) = tmpy_(iyn, ixn+(i-rfirst)*xn);
                const auto yChunks = (yn % Chunk<T>::size == 0) ? tmpy_.rowChunks() : tmpy_.rowChunks() - 1;
                for(long long yChunk = 0; yChunk < yChunks; yChunk++)
                  unaligned_store(tmpy_.chunk(yChunk, ixn+(i-rfirst)*xn), &y(yChunk*Chunk<T>::size+yn*ixn, i+iA*r));
                for(long long iyn = yChunks*Chunk<T>::size; iyn < yn; iyn++)
                  y(iyn+yn*ixn, i+iA*r) = tmpy_(iyn, ixn+(i-rfirst)*xn);
              }
          }
        }
        return;
      }

#pragma omp parallel for schedule(static)
      for(long long i = 0; i < r; i++)
        for(long long iA = 0; iA < rA; iA++)
        {
          const auto xstride = Eigen::OuterStride<>(&x(0,1) - &x(0,0));
          const auto ystride = Eigen::OuterStride<>(yn);
          const auto Astride = Eigen::OuterStride<>(&A(0,1) - &A(0,0));

          using map = Eigen::Map<mat, Eigen::Unaligned, Eigen::OuterStride<>>;
          using const_map = Eigen::Map<const mat, Eigen::Unaligned, Eigen::OuterStride<>>;

          const_map mapx(&x(i*xn,0), xn, m, xstride);
          const_map mapA(&A(0,m*iA), yn, m, Astride);
          map mapy(&y(0, i+iA*r), yn, xn, ystride);

#ifndef PITTS_DIRECT_GEMM
          mapy.noalias() = mapA * mapx.transpose();
#else
          cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasTrans, mapA.rows(), mapx.rows(), mapA.cols(), T(1), mapA.data(), mapA.colStride(), mapx.data(), mapx.colStride(), T(0), mapy.data(), mapy.colStride());
#endif
        }
    }
  }

  // implement TT Op apply to dense
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOp, const MultiVector<T>& MVx, MultiVector<T>& MVy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for matching dimensions
    const long long mTotal = std::accumulate(TTOp.column_dimensions().begin(), TTOp.column_dimensions().end(), 1, std::multiplies<long long>());
    if( mTotal != MVx.rows() )
      throw std::invalid_argument("TensorTrainOperator: input tensor dimension mismatch!");

    MultiVector<T> mv_tmp;

    // perform actual calculation
    const int nDim = TTOp.row_dimensions().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensor(iDim);
      const auto& x = (iDim == nDim-1) ? MVx : MVy;

      internal::apply_dense_contract(TTOp, iDim, subTOp, x, mv_tmp);

      std::swap(mv_tmp, MVy);
    }
  }


  template<typename T>
  TTOpApplyDenseHelper<T>::TTOpApplyDenseHelper(const TensorTrainOperator<T> &TTOp)
  {
    // only works for row-dims == col-dims so far
    if( TTOp.row_dimensions() != TTOp.column_dimensions() )
      throw std::invalid_argument("TTOpApplyDenseHelper: row-dims != col-dims not supported!");

    const auto timer = PITTS::timing::createScopedTimer<TTOpApplyDenseHelper<T>>();

    // get dimensions
    nDim_ = TTOp.row_dimensions().size();
    nTotal_ = std::accumulate(TTOp.row_dimensions().begin(), TTOp.row_dimensions().end(), 1, std::multiplies<long long>());
    nLast_ = TTOp.row_dimensions().back();
    const auto nChunks = (nTotal_/nLast_-1)/Chunk<T>::size + 1;
    const auto nChunksPadded = internal::paddedChunks(nChunks);
    nPadded_ = nChunksPadded * Chunk<T>::size;
    nTotalPadded_ = nPadded_* nLast_;

    // initialize requred data
    A_.resize(nDim_);
    rA_.resize(nDim_);
    r_.resize(nDim_);
    tmpv_.resize(nDim_+1);

    for(int iDim = nDim_-1; iDim >= 0; iDim--)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensor(iDim);
      rA_[iDim] = subTOp.r1(); 
      const auto ni = TTOp.row_dimensions()[iDim];
      const auto mi = TTOp.column_dimensions()[iDim];
      r_[(iDim + 1) % nDim_] = mi;
      A_[iDim].resize(ni, mi*subTOp.r2()*rA_[iDim]);
#pragma omp parallel for collapse(2) schedule(static)
      for(long long k2 = 0; k2 < subTOp.r2(); k2++)
        for(long long j = 0; j < mi; j++)
          for(long long i = 0; i < ni; i++)
            for(long long k1 = 0; k1 < rA_[iDim]; k1++)
              A_[iDim](i, j+k2*mi+k1*mi*subTOp.r2()) = subTOp(k1, TTOp.index(iDim, i, j), k2);
    }

    tmpX_.resize(omp_get_max_threads());
    tmpY_.resize(omp_get_max_threads());
  }

  template<typename T>
  void TTOpApplyDenseHelper<T>::addPadding(MultiVector<T> &x) const
  {
    if( x.cols() != 1 )
      throw std::invalid_argument("TTOpApplyDenseHelper: only works for MultiVectors with one column!");
    if( x.rows() != nTotal_ )
      throw std::invalid_argument("TTOpApplyDenseHelper: incorrect dimension on input to addPadding!");

    const auto timer = PITTS::timing::createScopedTimer<TTOpApplyDenseHelper<T>>();
    
    std::swap(x, tmpv(0));
    preparePadding(x);
    reshape(tmpv(0), nTotal_/nLast_, nLast_, x);
    assert(x.colStrideChunks()*Chunk<T>::size == nPadded_);

    x.resize(nTotalPadded_, 1, false, true);
  }

  template<typename T>
  void TTOpApplyDenseHelper<T>::removePadding(MultiVector<T> &y) const
  {
    if( y.cols() != 1 )
      throw std::invalid_argument("TTOpApplyDenseHelper: only works for MultiVectors with one column!");
    if( y.rows() != nTotalPadded_ )
      throw std::invalid_argument("TTOpApplyDenseHelper: incorrect dimension on input to removePadding!");

    const auto timer = PITTS::timing::createScopedTimer<TTOpApplyDenseHelper<T>>();

    y.resize(nTotal_/nLast_, nLast_, false, true);
    std::swap(y, tmpv(0));
    reshape(tmpv(0), nTotal_, 1, y);
  }

  template <typename T>
  inline void TTOpApplyDenseHelper<T>::preparePadding(MultiVector<T>& v) const
  {
    const auto timer = PITTS::timing::createScopedTimer<TTOpApplyDenseHelper<T>>();

    // ensure we have enough memory allocated to switch between n0 x N/n0 and nTotalPadded x 1
    v.resize(nTotalPadded_, 1, false);

    // for convenience resize to padded layout
    v.resize(nTotal_/nLast_, nLast_, false, true);
    // initialize padding in target memory to zero
#pragma omp parallel for collapse(2) schedule(static)
    for(long long j = 0; j < v.cols(); j++)
      for(long long iChunk = v.rowChunks()-1; iChunk < v.colStrideChunks(); iChunk++)
        v.chunk(iChunk, j) = Chunk<T>{};
  }

  // implement TT Op Helper apply to dense
  template<typename T>
  void apply(const TTOpApplyDenseHelper<T>& TTOp, MultiVector<T>& MVx, MultiVector<T>& MVy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TTOpApplyDenseHelper<T>>();

    // check for matching dimensions
    if( MVx.cols() != 1 )
      throw std::invalid_argument("TTOpApplyDenseHelper: only works for MultiVectors with one column!");
    if( MVx.rows() != TTOp.nTotalPadded() )
      throw std::invalid_argument("TTOpApplyDenseHelper: incorrect dimension on input to apply!");
  

    // ensure padding is correct in the result:
    TTOp.preparePadding(MVy);
    std::swap(MVy, TTOp.tmpv(0));

    // reuse memory of MVx (avoids copy), restored below
    MVx.resize(TTOp.nTotal()/TTOp.nLast(), TTOp.nLast(), false, true);
    std::swap(TTOp.tmpv(TTOp.nDim()), MVx);

    // perform actual calculation
    for(int iDim = TTOp.nDim()-1; iDim >= 0; iDim--)
      internal::apply_dense_contract_padded(TTOp.r(iDim), TTOp.rA(iDim), TTOp.tmpv(iDim+1), TTOp.A(iDim), TTOp.tmpv(iDim), TTOp.tmpX(), TTOp.tmpY());

    std::swap(TTOp.tmpv(0), MVy);
    MVy.resize(TTOp.nTotalPadded(), 1, false, true);
    std::swap(TTOp.tmpv(TTOp.nDim()), MVx);
    MVx.resize(TTOp.nTotalPadded(), 1, false, true);
  }

}

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_IMPL_HPP
