// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_symmetric_impl.hpp
* @brief Operations on symmetric tensors y = G S G^T given by a left half G (tensor train with an open right bond)
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-10-20
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SYMMETRIC_IMPL_HPP
#define PITTS_TENSORTRAIN_SYMMETRIC_IMPL_HPP

// includes
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "pitts_tensortrain_symmetric.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! operations on symmetric tensors G S G^T given by a left half G, see pitts_tensortrain_symmetric.hpp
    namespace symmetric_tt
    {
      // implement leftOrthonormalizeHalf
      template<typename T>
      Tensor2<T> leftOrthonormalizeHalf(TensorTrain<T>& H)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int m = H.dimensions().size();
        if( m > 1 )
          leftNormalize_range(H, 0, m-1, T(0));

        // the last core with its open bond: QR of the left unfolding
        const auto& last = H.subTensor(m-1);
        auto [Q, R] = normalize_qb(unfold_left(last), true);
        H.setSubTensor(m-1, fold_left(std::move(Q), last.n()), TT_Orthogonality::left);
        return std::move(R);
      }


      // implement halfInnerProducts
      template<typename T>
      Tensor2<T> halfInnerProducts(const TensorTrain<T>& A, const TensorTrain<T>& B)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        if( A.dimensions() != B.dimensions() )
          throw std::invalid_argument("halfInnerProducts: dimensions do not match!");

        Tensor2<T> E(1, 1);
        E(0, 0) = T(1);
        Tensor3<T> EB;
        for(int k = 0; k < int(A.dimensions().size()); k++)
        {
          // E_{k+1} = A_k(*,*,:)^T (E_k B_k)(*,*,:)
          normalize_contract1(E, B.subTensor(k), EB);
          reverse_dot_contract2(A.subTensor(k), EB, E);
        }
        return E;
      }


      // implement rightOrthonormalHalf
      template<typename T>
      void rightOrthonormalHalf(const TensorTrain<T>& G, const Tensor2<T>& S, TensorTrain<T>& Y)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int m = G.dimensions().size();
        if( S.r1() != G.subTensor(m-1).r2() )
          throw std::invalid_argument("rightOrthonormalHalf: dimensions of G and S do not match!");
        if( Y.dimensions() != G.dimensions() )
          Y = TensorTrain<T>(G.dimensions());
        copy(G, Y);
        Tensor3<T> lastTimesS;
        normalize_contract2(G.subTensor(m-1), S, lastTimesS);
        Y.setSubTensor(m-1, std::move(lastTimesS));
        if( m > 1 )
          rightNormalize_range(Y, 0, m-1, T(0));
      }


      // implement symmetricEigen
      template<typename T>
      SymmetricEigen<T> symmetricEigen(const Tensor2<T>& C, T rankTolerance, int maxRank, bool absoluteTolerance)
      {
        static_assert(std::is_floating_point_v<T>, "symmetricEigen: only for real types");

        const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

        const int n = C.r1();
        if( C.r2() != n )
          throw std::invalid_argument("symmetricEigen: matrix is not square!");

        const Eigen::MatrixX<T> Csym = (ConstEigenMap(C) + ConstEigenMap(C).transpose()) / T(2);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<T>> eig(Csym);

        // order by decreasing |lambda|
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int i, int j){return std::abs(eig.eigenvalues()(i)) > std::abs(eig.eigenvalues()(j));});

        // truncation in the Frobenius norm as for the SVD (the singular values of C are the |lambda|), at most maxRank
        Eigen::VectorX<T> absLambda(n);
        for(int i = 0; i < n; i++)
          absLambda(i) = std::abs(eig.eigenvalues()(order[i]));
        const T tolerance = absoluteTolerance ? rankTolerance : rankTolerance * Csym.norm();
        const int rank = std::max(1, std::min(rankInFrobeniusNorm(absLambda, n, tolerance), maxRank));

        SymmetricEigen<T> result;
        result.U.resize(n, rank);
        result.lambda.resize(rank);
        for(int j = 0; j < rank; j++)
        {
          EigenMap(result.U).col(j) = eig.eigenvectors().col(order[j]);
          result.lambda[j] = eig.eigenvalues()(order[j]);
        }
        return result;
      }


      // implement symmetricRound
      template<typename T>
      void symmetricRound(TensorTrain<T>& G, Tensor2<T>& S, T rankTolerance, int maxRank)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int m = G.dimensions().size();
        const int r = S.r1();
        if( S.r2() != r || G.subTensor(m-1).r2() != r )
          throw std::invalid_argument("symmetricRound: dimensions of G and S do not match!");

        // tolerance per bond of y, see the error bound in the header
        const T tau = rankTolerance / std::sqrt(T(2*m-1));
        const T yNorm = ConstEigenMap(S).norm();   // ||y||_F = ||S||_F for a left-orthonormal G

        // 1. T = G S, right-orthonormal cores 1..m-1, then truncation of the inner bonds from left to right
        TensorTrain<T> half(G.dimensions());
        rightOrthonormalHalf(G, S, half);
        // leftNormalize_range uses rankTolerance / sqrt(m-1) per bond, relative to the norm of the first core (= ||y||)
        if( m > 1 )
          leftNormalize_range(half, 0, m-1, T(tau * std::sqrt(T(m-1))), maxRank);
        // the last core: left-orthonormal factor and the remainder M = G'^T T
        const auto& last = half.subTensor(m-1);
        auto [Q, M] = normalize_qb(unfold_left(last), true);
        half.setSubTensor(m-1, fold_left(std::move(Q), last.n()), TT_Orthogonality::left);

        // 2. symmetric projection: S'' = M W^T with W = G'^T G
        const Tensor2<T> W = halfInnerProducts(half, G);
        Tensor2<T> S2(M.r1(), W.r1());
        EigenMap(S2) = ConstEigenMap(M) * ConstEigenMap(W).transpose();

        // 3. middle bond: truncated eigendecomposition, the eigenvectors are multiplied into the last core
        const SymmetricEigen<T> eig = symmetricEigen(S2, T(tau * yNorm), maxRank, true);
        Tensor3<T> newLast;
        normalize_contract2(half.subTensor(m-1), eig.U, newLast);
        half.setSubTensor(m-1, std::move(newLast), TT_Orthogonality::left);

        const int rNew = eig.lambda.size();
        S.resize(rNew, rNew);
        EigenMap(S).setZero();
        for(int i = 0; i < rNew; i++)
          S(i, i) = eig.lambda[i];
        G = std::move(half);
      }


      // implement positiveHalf
      template<typename T>
      T positiveHalf(const TensorTrain<T>& G, const Tensor2<T>& S, TensorTrain<T>& H)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int m = G.dimensions().size();
        if( S.r1() != S.r2() || G.subTensor(m-1).r2() != S.r1() )
          throw std::invalid_argument("positiveHalf: dimensions of G and S do not match!");

        const SymmetricEigen<T> eig = symmetricEigen(S);
        std::vector<int> positive;
        T negativeNorm = 0;
        for(int j = 0; j < int(eig.lambda.size()); j++)
        {
          if( eig.lambda[j] > 0 )
            positive.push_back(j);
          else
            negativeNorm += eig.lambda[j] * eig.lambda[j];
        }

        // factor U_+ diag(sqrt(lambda_+)), a zero column if there is no positive eigenvalue
        Tensor2<T> F(S.r1(), std::max<int>(positive.size(), 1));
        EigenMap(F).setZero();
        for(int j = 0; j < int(positive.size()); j++)
          EigenMap(F).col(j) = std::sqrt(eig.lambda[positive[j]]) * ConstEigenMap(eig.U).col(positive[j]);

        if( H.dimensions() != G.dimensions() )
          H = TensorTrain<T>(G.dimensions());
        copy(G, H);
        Tensor3<T> last;
        normalize_contract2(G.subTensor(m-1), F, last);
        H.setSubTensor(m-1, std::move(last));
        return std::sqrt(negativeNorm);
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_SYMMETRIC_IMPL_HPP
