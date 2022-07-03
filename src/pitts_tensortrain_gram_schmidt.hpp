/*! @file pitts_tensortrain_gram_schmidt.hpp
* @brief Modified and iterated Gram-Schmidt for orthogonalizing vectors in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-02
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP
#define PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP

// includes
#include <iostream>
#include <cassert>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_timer.hpp"
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! index of maximal absolute value of an Eigen vector
    template<typename T>
    auto argmaxabs(const Eigen::Array<T, Eigen::Dynamic, 1>& v)
    {
      if( v.size() == 0 )
        throw std::invalid_argument("Called argmax for an empty array");
      
      int idx = 0;
      for(int i = 1; i < v.size(); i++)
        if( std::abs(v(i)) > std::abs(v(idx)) )
          idx = i;

      return idx;
    }

    //! get max rank of a tensor train
    template<typename T>
    int maxRank(const TensorTrain<T>& TT)
    {
      int max_r = 0;
      for(auto r: TT.dimensions())
        max_r = std::max(r, max_r);
      return max_r;
    }

  }

  //! Modified Gram-Schmidt orthogonalization algorithm in Tensor-Train format
  //!
  //! Orthogonalizes w wrt V=(v_1, ..., v_k) and normalizes w.
  //! Then adds w to the list of directions V.
  //!
  //! The modified Gram-Schmidt process (MGS) is adopted to the special behavior of tensor-train arithmetic:
  //! Per default uses pivoting (more dot calls), multiple iterations (to increase robustness/accuracy), and skips directions v_i where <v_i,w> is already small.
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param V                orthogonal directions in TT format, orthogonalized w is appended
  //! @param w                new direction in TT format, orthogonalized wrt. V
  //! @param rankTolerance    desired approximation accuracy (for TT axpby / normalize)
  //! @param maxRank          maximal allowed approximation rank (for TT axpby / normalize)
  //! @param outputPrefix     string to prefix all output about the convergence history
  //! @param verbose          set to true, to print the residual norm in each iteration to std::cout
  //! @param nIter            number of iterations for iterated Gram-Schmidt
  //! @param pivoting         enable or disable pivoting (enabled per default)
  //! @param modified         use modified Gram-Schmidt: recalculate <v_i,w> (enabled per default)
  //! @param skipDirs         skip axpy operations, when <v_i,w> already < tolerance
  //! @return                 dot-products <v_1,w> ... <v_k,w> and norm of w after orthog. wrt. V (||w-VV^Tw||)
  //!
  template<typename T>
  auto gram_schmidt(std::vector<TensorTrain<T>>& V, TensorTrain<T>& w,
                    T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max(),
                    const std::string& outputPrefix = "", bool verbose = false,
                    int nIter = 4, bool pivoting = true, bool modified = true, bool skipDirs = true)
  {
    using arr = Eigen::Array<T, Eigen::Dynamic, 1>;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nV = V.size();

    T alpha = normalize(w, rankTolerance, maxRank);
    arr h = arr::Zero(nV+1);
    if( nV == 0 )
    {
      h(0) = alpha;
      V.emplace_back(std::move(w));
      return h;
    }

    arr Vtw(nV);
    for(int iter = 0; iter < nIter; iter++)
    {
      if( pivoting || (!modified) )
      {
        for (int i = 0; i < nV; i++)
          Vtw(i) = dot(V[i], w);

        const T maxErr = Vtw.abs().maxCoeff();
        if( verbose )
          std::cout << outputPrefix << "orthog. max. error: " << maxErr << ", w max. rank: " << internal::maxRank(w) << "\n";
        if( maxErr < rankTolerance )
          break;
      }

      for(int i = 0; i < nV; i++)
      {
        int pivot = i;
        if( pivoting )
          pivot = internal::argmaxabs(Vtw);
        T beta = Vtw(pivot);
        if( (!pivoting) && modified )
          beta = dot(V[pivot], w);

        if( skipDirs && std::abs(beta) < rankTolerance )
        {
          if( pivoting )
            break;
          else
            continue;
        }

        if( pivoting && modified && i > 0 )
          beta = dot(V[pivot], w);
        
        h(pivot) += alpha * beta;
        alpha = alpha * axpby(-beta, V[pivot], T(1), w, rankTolerance, maxRank);
        Vtw(pivot) = T(0);
      }
    }

    V.emplace_back(std::move(w));
    h(nV) = alpha;
    return h;
  }

}


#endif // PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP
