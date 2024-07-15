// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_common.hpp"
#include "pitts_mkl.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>
#include <cassert>

// pitts_tensor3_split_impl.hpp internal::normalize_svd_only

namespace
{
  using Type = double;
  
  using EigenMatrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;

#if EIGEN_VERSION_AT_LEAST(3,4,90)
  using EigenSVD = Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU>;
#else
  using EigenSVD = Eigen::BDCSVD<EigenMatrix>;
#endif

  struct Work
  {
    EigenSVD svd;
    PITTS::Tensor2<Type> tmpA, S, U, Vt;
  };


  //! small wrapper around SVD, avoid reallocating svd data each time...
  void calculate_svd(const PITTS::Tensor2<Type>& M, Work& w)
  {
    using namespace PITTS;

    const auto n = M.r1();
    const auto m = M.r2();

#ifdef PITTS_DIRECT_MKL_GEMM
    copy(M, w.tmpA);
    w.S.resize(std::min(n,m),1);
    w.U.resize(n, std::min(n,m));
    w.Vt.resize(std::min(n,m), m);
#endif


    constexpr auto pow3 = [](double x){return x*x*x;};

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<Type>>(
      {{"n", "m"},{n, m}}, // arguments
      {{(4.*n*m*std::min(n,m)+2*pow3(std::min(n,m))/3)*kernel_info::FMA<Type>()}, // flops
       {(n*m)*kernel_info::Load<Type>() + ((n+m+1)*std::min(n,m))*kernel_info::Store<Type>()}} // data transfers
      );

    //std::cout << "SVD of:\n" << ConstEigenMap(M) << std::endl;
    const auto mapM = ConstEigenMap(M);

#ifndef PITTS_DIRECT_MKL_GEMM

#if EIGEN_VERSION_AT_LEAST(3,4,90)
    w.svd.compute(mapM);
#else
    w.svd.compute(mapM, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

    assert(!std::isnan(w.svd.singularValues()(0)));
#else
    LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.S(0,0), &w.U(0,0), w.U.r1(), &w.Vt(0,0), w.Vt.r1());

    // check result
    //std::cout << "singular values: " << ConstEigenMap(w.S).transpose() << "\n";
    //std::cout << "error: " << (ConstEigenMap(w.U) * ConstEigenMap(w.S).asDiagonal() * ConstEigenMap(w.Vt) - ConstEigenMap(M)).norm() << "\n";
#endif
  }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 4 )
    throw std::invalid_argument("Requires 4 arguments (n m nIter)!");

  long long n = 0, m = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);

  using Type = double;
  PITTS::Tensor2<Type> M(n,m);
  randomize(M);
  for(int i = 0; i < std::min(n,m); i++)
    M(i,i) = 5 + 0.1*M(i,i);

  Work w;
  calculate_svd(M, w);

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    calculate_svd(M, w);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "SVD wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
