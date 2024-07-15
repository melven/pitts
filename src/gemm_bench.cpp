// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <charconv>


namespace
{
  using namespace PITTS;
  template<typename T>
  void gemm_unpadded(const Tensor2<T>& A, const Tensor2<T>& B, Tensor2<T>& C)
  {
    // check dimensions
    if( A.r2() != B.r1() || A.r1() != C.r1() || C.r2() != B.r2() )
      throw std::invalid_argument("gemm: dimension mismatch!");

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"n", "m", "k"},{A.r1(),A.r2(),C.r2()}}, // arguments
        {{(double(A.r1())*B.r1()*B.r2())*kernel_info::FMA<T>()}, // flops
         {(double(A.r1())*A.r2() + B.r1()*B.r2())*kernel_info::Load<T>() + (double(A.r1())*B.r2())*kernel_info::Store<T>()}} // data transfers
        );
    
#ifndef PITTS_DIRECT_MKL_GEMM
    EigenMap(C).noalias() = ConstEigenMap(A) * ConstEigenMap(B);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.r1(), B.r2(), A.r2(), 1., &A(0,0), A.r1(), &B(0,0), B.r1(), 0., &C(0,0), C.r1());
#endif
  }

  using namespace PITTS;
  template<typename T>
  void gemm_padded(const MultiVector<T>& X, const MultiVector<T>& Y, MultiVector<T>& Z)
  {
    // check dimensions
    if( X.cols() != Y.rows() || X.rows() != Z.rows() || Z.cols() != Y.cols() )
      throw std::invalid_argument("gemm_padded: dimension mismatch!");

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"n", "m", "k"},{X.rows(),X.cols(),Z.cols()}}, // arguments
        {{(double(X.rows())*Y.rows()*Y.cols())*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*X.cols() + Y.rows()*Y.cols())*kernel_info::Load<T>() + (double(X.rows())*Y.cols())*kernel_info::Store<T>()}} // data transfers
        );
    
#ifndef PITTS_DIRECT_MKL_GEMM
    EigenMap(Z).noalias() = ConstEigenMap(X) * ConstEigenMap(Y);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, X.rows(), Y.cols(), X.cols(), 1., &X(0,0), X.colStrideChunks()*Chunk<T>::size, &Y(0,0), Y.colStrideChunks()*Chunk<T>::size, 0., &Z(0,0), Z.colStrideChunks()*Chunk<T>::size);
#endif
  }

}



int main(int argc, char* argv[])
{
  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n m k nIter)!");

  int n = 0, m = 0, k = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIter);

  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::MultiVector<Type> mvX(n,m), mvY(m,k), mvZ(n,k);
  randomize(mvX);
  randomize(mvY);
  randomize(mvZ);

  PITTS::Tensor2<Type> t2A(n,m), t2B(m,k), t2C(n,k);

  // allow MKL to initialize stuff (slow/costly)
  gemm_unpadded(t2A, t2B, t2C);
  gemm_unpadded(t2A, t2B, t2C);

  gemm_padded(mvX, mvY, mvZ);
  gemm_padded(mvX, mvY, mvZ);

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();

  for(int iter = 0; iter < nIter; iter++)
    gemm_unpadded(t2A, t2B, t2C);

  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "gemm_unpadded: " << wtime << std::endl;


  wtime = omp_get_wtime();

  for(int iter = 0; iter < nIter; iter++)
    gemm_padded(mvX, mvY, mvZ);

  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "gemm_padded: " << wtime << std::endl;


  PITTS::finalize();

  return 0;
}
