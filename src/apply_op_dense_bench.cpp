// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_reshape.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include <iostream>
#include <charconv>


namespace
{
  using namespace PITTS;
  template<typename T>
  void gemm(const MultiVector<T>& X, const Tensor2<T>& M, MultiVector<T>& Y)
  {
    // check dimensions
    if( X.cols() != M.r1() || X.rows() != Y.rows() || Y.cols() != M.r2() )
      throw std::invalid_argument("MultiVector::transform: dimension mismatch!");

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"n", "m", "k"},{X.rows(),X.cols(),Y.cols()}}, // arguments
        {{(double(X.rows())*M.r1()*M.r2())*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*X.cols() + M.r1()*M.r2())*kernel_info::Load<T>() + (double(X.rows())*M.r2())*kernel_info::Store<T>()}} // data transfers
        );
    
#ifndef PITTS_DIRECT_MKL_GEMM
    EigenMap(Y).noalias() = ConstEigenMap(X) * ConstEigenMap(M);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, X.rows(), M.r2(), X.cols(), 1., &X(0,0), X.colStrideChunks()*Chunk<T>::size, &M(0,0), M.r1(), 0., &Y(0,0), Y.colStrideChunks()*Chunk<T>::size);
#endif
  }

  //! assumes a very special memory layout: A(:,(*,1:rA)) * x(:,(:,*))^T -> y(:,(:,1:rA))
  template<typename T>
  void apply_dense_contract(long long r, const MultiVector<T>& x, const MultiVector<T>& A, MultiVector<T>& y)
  {
    const auto yn = A.rows();
    const auto xn = x.rows();
    const auto m = x.cols() / r;
    const auto rA = A.cols() / m;
    if( x.cols() % r != 0 || A.cols() % m != 0 )
      throw std::invalid_argument("apply_dense_contract: invalid dimensions!");
    
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"yn", "xn", "m", "rA", "r"},{yn, xn, m, rA, r}}, // arguments
        {{double(xn*r*m*yn*rA)*kernel_info::FMA<T>()}, // flops
         {double(xn*r*m + yn*m*rA)*kernel_info::Load<T>() + double(yn*xn*r*rA)*kernel_info::Store<T>()}} // data transfers
        );
    
    y.resize(yn, xn*r*rA);

#pragma omp parallel for collapse(2) schedule(static)
    for(long long iA = 0; iA < rA; iA++)
      for(long long i = 0; i < r; i++)
      {
        const auto xstride = Eigen::OuterStride<>(&x(0,r) - &x(0,0));
        const auto ystride = Eigen::OuterStride<>(&y(0,1) - &y(0,0));

        using mat = Eigen::MatrixX<T>;
        using map = Eigen::Map<mat, EigenAligned, Eigen::OuterStride<>>;
        using const_map = Eigen::Map<const mat, EigenAligned, Eigen::OuterStride<>>;

        const_map mapx(&x(0,i), xn, m, xstride);
        const_map mapA(&A(0,m*iA), yn, m, ystride);
        map mapy(&y(0,i*xn+iA*xn*r), yn, xn, ystride);

        mapy.noalias() = mapA * mapx.transpose();
      }
  }

}



int main(int argc, char* argv[])
{
  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n r rOp nIter)!");

  int n = 0, r = 0, rOp = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], r);
  std::from_chars(argv[3], argv[4], rOp);
  std::from_chars(argv[4], argv[5], nIter);

  const std::vector<int> dims = {r,n,r};
  const long long N = (long long)(r) * (long long)(n) * (long long)(r);

  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrainOperator<Type> TTOp(dims, dims, rOp);
  randomize(TTOp.tensorTrain());
  PITTS::MultiVector<Type> mvX(N,1), mvY(N,1);
  randomize(mvX);
  randomize(mvY);

  for(int iter = 0; iter < nIter; iter++)
    apply(TTOp, mvX, mvY);
  

  // special padded / faster variant
  PITTS::TTOpApplyDenseHelper TTOpHelper(TTOp);
  TTOpHelper.addPadding(mvX);
  TTOpHelper.addPadding(mvY);

  for(int iter = 0; iter < nIter; iter++)
    apply(TTOpHelper, mvX, mvY);
  
  TTOpHelper.removePadding(mvX);
  TTOpHelper.removePadding(mvY);
  
#if 0

  // required GEMMs
  mvX.resize(r*n,r);
  mvY.resize(r*n,r*rOp);
  PITTS::Tensor2<Type> A(r,r*rOp);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);

  mvX.resize(r*r,n*rOp);
  mvY.resize(r*r,n*rOp);
  A.resize(n*rOp,n*rOp);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);

  mvX.resize(r*n,r*rOp);
  mvY.resize(r*n,r);
  A.resize(r*rOp,r);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);


  const auto r1 = r;
  const auto r2 = r;
  const auto rA1 = rOp;
  const auto rA2 = rOp;
  // specialized contractions
  //      X                A3              tmp1
  // (r1 x n _r2) x (r2 x _r2 rA2) -> (r2 x r1 n rA2)
  //      tmp1                  A2                 tmp2
  // (r2 x r1 _n _rA2) x (n x _n _rA2 rA1) -> (n x r2 r1 rA1)
  //      tmp2                  A1                 y
  // (n x r2 _r1 _rA1) x (r1 x _r1 _rA1) -> (r1 x n r2)
  mvX.resize(r1, n*r2);
  MultiVector<Type> A3(r2, r2*rA2), A2(n,n*rA2*rA1), A1(r1, r1*rA1);
  MultiVector<Type> tmp1(r2, r1*n*rA2), tmp2(n, r2*r1*rA1);
  mvY.resize(r1, n*r2);
  randomize(mvX);
  randomize(mvY);
  randomize(A3);
  randomize(A2);
  randomize(A1);
  randomize(tmp1);
  randomize(tmp2);

  //for(int iter = 0; iter < nIter; iter++)
  //{
  //  apply_dense_contract(n, mvX, A3, tmp1);
  //  apply_dense_contract(r1, tmp1, A2, tmp2);
  //  apply_dense_contract(r2, tmp2, A1, mvY);
  //}

#endif

  PITTS::finalize();

  return 0;
}
