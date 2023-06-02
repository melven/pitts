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
    
    EigenMap(Y).noalias() = ConstEigenMap(X) * ConstEigenMap(M);
  }

  constexpr unsigned long long r1 = 330;
  constexpr unsigned long long n = 50;
  constexpr unsigned long long r2 = 330;
  constexpr unsigned long long rA1 = 2;
  constexpr unsigned long long rA2 = 2;

  // (r1 x n _r2) x (r2 x _r2 rA2) -> (r2 x r1 n rA2)
  template<typename T>
  void contract1(const MultiVector<T>& x, const MultiVector<T>& A3, MultiVector<T>& y1)
  {
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"r1", "n", "r2", "rA2"},{r1,n,r2,rA2}}, // arguments
        {{double(r1*n*r2*r2*rA2)*kernel_info::FMA<T>()}, // flops
         {double(r1*n*r2 + r2*r2*rA2)*kernel_info::Load<T>() + double(r2*r1*n*rA2)*kernel_info::Store<T>()}} // data transfers
        );
    

#pragma omp parallel for collapse(2) schedule(static)
    for(unsigned long long iA2 = 0; iA2 < rA2; iA2++)
      for(unsigned long long in = 0; in < n; in++)
      {
        const auto xStride = &x(0,n) - &x(0,0);
        const auto A3Stride = &A3(0,1) - &A3(0,0);
        using mat = Eigen::MatrixX<T>;
        using const_map = Eigen::Map<const mat, Eigen::Aligned128, Eigen::OuterStride<>>;
        using map = Eigen::Map<mat, Eigen::Aligned128, Eigen::OuterStride<>>;
        const_map mapX(&x(0,in), r1, r2, Eigen::OuterStride<>(xStride));
        const_map mapA3(&A3(0,r2*iA2), r2, r2, Eigen::OuterStride<>(A3Stride));
        map mapY1(&y1(0,r1*in + r1*n*iA2), r2, r1, Eigen::OuterStride<>(A3Stride));

        mapY1.noalias() = mapA3 * mapX.transpose();
      }
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
        using map = Eigen::Map<mat, Eigen::Aligned128, Eigen::OuterStride<>>;
        using const_map = Eigen::Map<const mat, Eigen::Aligned128, Eigen::OuterStride<>>;

        const_map mapx(&x(0,i), xn, m, xstride);
        const_map mapA(&A(0,m*iA), yn, m, ystride);
        map mapy(&y(0,i*xn+iA*xn*r), yn, xn, ystride);

        mapy.noalias() = mapA * mapx.transpose();
      }
  }

  // (r2 x r1 _n _rA2) x (n x _n _rA2 rA1) -> (n x r2 r1 rA1)
  template<typename T>
  void contract2(const MultiVector<T>& y1, const MultiVector<T>& A2, MultiVector<T>& y2)
  {
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"r1", "n", "r2", "rA1", "rA2"},{r1,n,r2,rA1,rA2}}, // arguments
        {{double(r1*r1*n*rA2*n*rA1)*kernel_info::FMA<T>()}, // flops
         {double(r2*r1*n*rA2 + n*n*rA2*rA1)*kernel_info::Load<T>() + double(n*r2*r1*rA1)*kernel_info::Store<T>()}} // data transfers
        );
    
#pragma omp parallel for collapse(2) schedule(static)
    for(unsigned long long iA1 = 0; iA1 < rA1; iA1++)
      for(unsigned long long i1 = 0; i1 < r1; i1++)
      {
        const auto y1Stride = &y1(0,r1) - &y1(0,0);
        const auto A2Stride = &A2(0,1) - &A2(0,0);
        using mat = Eigen::MatrixX<T>;
        using const_map = Eigen::Map<const mat, Eigen::Aligned128, Eigen::OuterStride<>>;
        using map = Eigen::Map<mat, Eigen::Aligned128, Eigen::OuterStride<>>;
        const_map mapY1(&y2(0,i1), r2, n*rA2, Eigen::OuterStride<>(y1Stride));
        const_map mapA2(&A2(0,n*rA2*iA1), n, n*rA2, Eigen::OuterStride<>(A2Stride));
        map mapY2(&y2(0,r2*i1+r2*r2*iA1), n, r2, Eigen::OuterStride<>(A2Stride));

        mapY2.noalias() = mapA2 * mapY1.transpose();
      }
  }

  // (n x r2 _r1 _rA1) x (r1 x _r1 _rA1) -> (r1 x n r2)
  template<typename T>
  void contract3(const MultiVector<T>& y2, const MultiVector<T>& A1, MultiVector<T>& y)
  {
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"r1", "n", "r2", "rA1"},{r1,n,r2,rA1}}, // arguments
        {{double(n*r2*r1*rA1*r2)*kernel_info::FMA<T>()}, // flops
         {double(n*r2*r1*rA1 + r1*r1*rA1)*kernel_info::Load<T>() + double(r1*n*r2)*kernel_info::Store<T>()}} // data transfers
        );
    
#pragma omp parallel for schedule(static)
    for(unsigned long long i2 = 0; i2 < r2; i2++)
    {
      const auto y2Stride = &y2(0,r2) - &y2(0,0);
      const auto A1Stride = &A1(0,1) - &A1(0,0);
      using mat = Eigen::MatrixX<T>;
      using const_map = Eigen::Map<const mat, Eigen::Aligned128, Eigen::OuterStride<>>;
      using map = Eigen::Map<mat, Eigen::Aligned128, Eigen::OuterStride<>>;
      const_map mapY2(&y2(0,i2), n, r1*rA1, Eigen::OuterStride<>(y2Stride));
      const_map mapA1(&A1(0,0), r1, r1*rA1, Eigen::OuterStride<>(A1Stride));
      map mapY(&y(0,n*i2), r1, n, Eigen::OuterStride<>(A1Stride));

      mapY.noalias() = mapA1 * mapY2.transpose();
    }
  }

}



int main(int argc, char* argv[])
{
  const std::vector dims = {330,50,330};
  const long long N = 330*50*330;
  const int rOp = 2;
  const int nIter = 100;

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
  

  // required GEMMs
  mvX.resize(330*50,330);
  mvY.resize(330*50,330*2);
  PITTS::Tensor2<Type> A(330,330*2);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);

  mvX.resize(330*330,50*2);
  mvY.resize(330*330,50*2);
  A.resize(50*2,50*2);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);

  mvX.resize(330*50,330*2);
  mvY.resize(330*50,330);
  A.resize(330*2,330);
  randomize(mvX);
  randomize(mvY);
  randomize(A);

  for(int iter = 0; iter < nIter; iter++)
    transform(mvX, A, mvY);
  for(int iter = 0; iter < nIter; iter++)
    gemm(mvX, A, mvY);


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

  for(int iter = 0; iter < nIter; iter++)
  {
    contract1(mvX, A3, tmp1);
    contract2(tmp1, A2, tmp2);
    contract3(tmp2, A1, mvY);
  }

  for(int iter = 0; iter < nIter; iter++)
  {
    apply_dense_contract(n, mvX, A3, tmp1);
    apply_dense_contract(r1, tmp1, A2, tmp2);
    apply_dense_contract(r2, tmp2, A1, mvY);
  }


  PITTS::finalize();

  return 0;
}
