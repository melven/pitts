#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"
#include <iostream>

namespace
{
  auto& operator<<(auto &stream, const std::vector<int>& vec)
  {
    for(auto v: vec)
      stream << v << " ";
    return stream;
  }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> rhs(5,17);
  rhs.setOnes();

  // simple CG algorithm
  PITTS::TensorTrain<Type> x(rhs.dimensions());
  x.setZero();
  auto xnorm = Type(0);
  PITTS::TensorTrain<Type> r = rhs;
  auto rnorm = normalize(r);
  PITTS::TensorTrain<Type> p = r;
  auto pnorm = rnorm;
  PITTS::TensorTrain<Type> q(p.dimensions());
  const auto maxIter = 50;
  const auto resTol = 1.e-3*rnorm;
  const auto rankTol = 1.e-12;
  for(int iter = 0; iter < maxIter; iter++)
  {
    // q = Ap
    copy(p, q); const auto qnorm = -pnorm * laplaceOperator(q, rankTol);

    // p^TAp = q^Tp
    const auto qTp = qnorm * pnorm * dot(q,p);

    // alpha = rTr / qTp
    const auto alpha = rnorm*rnorm / qTp;

    // x = x + alpha*p
    xnorm = axpby(alpha*pnorm, p, xnorm, x, rankTol);

    // r = r - alpha*q
    const auto old_rnorm = rnorm;
    rnorm = axpby(-alpha * qnorm, q, rnorm, r, rankTol);
    std::cout << "Residual norm: " << rnorm << std::endl;
    if( rnorm < resTol )
      break;

    // beta = rTr / old_rTr
    const auto beta = rnorm*rnorm / (old_rnorm*old_rnorm);

    // p = p + beta*r
    pnorm = axpby(rnorm, r, beta*pnorm, p, rankTol);
  }
  // calculate real residual
  PITTS::TensorTrain<Type> y = x;
  const auto ynorm = -xnorm*laplaceOperator(y);
  PITTS::TensorTrain<Type> r_ref = y;
  const auto rnorm_ref = axpby(1., rhs, -ynorm, r_ref);
  std::cout << "Real residual norm: " << rnorm_ref << std::endl;
  std::cout << "x TTranks: " << x.getTTranks() << std::endl;
  std::cout << "r TTranks: " << r.getTTranks() << std::endl;
  
  // try "transpose" + normalize
  const auto nDim = x.dimensions().size();
  std::vector<PITTS::Tensor3<Type>> subT_xT(nDim);
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    const int jDim = nDim - 1 - iDim;
    const auto& oldSubT = x.subTensor(iDim);
    const auto r1 = oldSubT.r2();
    const auto n = oldSubT.n();
    const auto r2 = oldSubT.r1();
    subT_xT[jDim].resize(r1,n,r2);
    for(int j = 0; j < r2; j++)
      for(int k = 0; k < n; k++)
        for(int i = 0; i < r1; i++)
          subT_xT[jDim](i,k,j) = oldSubT(j,k,i);
  }
  PITTS::TensorTrain<Type> xT(std::move(subT_xT));
  const auto xTnorm = normalize(xT, rankTol);
  std::cout << "xTnorm: " << xTnorm << std::endl;
  std::cout << "xT TTranks: " << xT.getTTranks() << std::endl;

  PITTS::finalize();

  return 0;
}
