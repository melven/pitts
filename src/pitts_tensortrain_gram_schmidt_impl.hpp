/*! @file pitts_tensortrain_gram_schmidt_impl.hpp
* @brief Modified and iterated Gram-Schmidt for orthogonalizing vectors in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-02
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_GRAM_SCHMIDT_IMPL_HPP
#define PITTS_TENSORTRAIN_GRAM_SCHMIDT_IMPL_HPP

// includes
#include <iostream>
#include <cassert>
#include "pitts_tensortrain_gram_schmidt.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! index of maximal absolute value of an Eigen vector
    template<typename T>
    auto argmaxabs(const Eigen::ArrayX<T>& v)
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
      for(auto r: TT.getTTranks())
        max_r = std::max(r, max_r);
      return max_r;
    }

  }

  // implement TT gramSchmidt
  template<typename T>
  Eigen::ArrayX<T> gramSchmidt(std::vector<TensorTrain<T>>& V, TensorTrain<T>& w,
                               T rankTolerance, int maxRank, bool symmetric,
                               const std::string_view& outputPrefix, bool verbose,
                               int nIter, bool pivoting, bool modified, bool skipDirs)
  {
    using arr = Eigen::ArrayX<T>;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nV = V.size();
    // adjust first vector in V to consider for symmetric cases
    const int firstV = symmetric ? std::max(0, nV-2) : 0;

    T alpha = normalize(w, rankTolerance, maxRank);
    arr h = arr::Zero(nV+1);
    if( nV == 0 )
    {
      h(0) = alpha;
      V.emplace_back(std::move(w));
      return h;
    }

    arr Vtw = arr::Zero(nV);
    for(int iter = 0; iter < nIter; iter++)
    {
      if( pivoting || (!modified) )
      {
        for (int i = firstV; i < nV; i++)
          Vtw(i) = dot(V[i], w);

        const T maxErr = Vtw.abs().maxCoeff();
        if( verbose )
          std::cout << outputPrefix << "orthog. max. error: " << maxErr << ", w max. rank: " << internal::maxRank(w) << "\n";
        if( maxErr < rankTolerance )
          break;
      }

      for(int i = firstV; i < nV; i++)
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

    if( verbose )
    {
      if( (!pivoting) && modified )
      {
        for (int i = firstV; i < nV; i++)
          Vtw(i) = dot(V[i], w);

        const T maxErr = Vtw.abs().maxCoeff();
        std::cout << outputPrefix << "orthog. max. error: " << maxErr << ", w max. rank: " << internal::maxRank(w) << "\n";
      }
    }

    V.emplace_back(std::move(w));
    h(nV) = alpha;
    return h;
  }

}


#endif // PITTS_TENSORTRAIN_GRAM_SCHMIDT_IMPL_HPP
