/*! @file pitts_tensor3_split.hpp
* @brief split a simple rank-3 tensor into two (using a QR decomposition)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_SPLIT_HPP
#define PITTS_TENSOR3_SPLIT_HPP

// includes
#include <limits>
#include "pitts_tensor3.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! wrapper for qr, allows to show timings, returns LQ decomposition for leftOrthog=false
    template<typename T>
    std::pair<Tensor2<T>,Tensor2<T>> normalize_qb(const ConstTensor2View<T>& M, bool leftOrthog = true, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max(), bool absoluteTolerance = false);
    

    //! wrapper for truncated SVD, allows to show timings, directly combines singular values with lefT/right singular vectors
    template<typename T>
    std::pair<Tensor2<T>,Tensor2<T>> normalize_svd(const ConstTensor2View<T>& M, bool leftOrthog, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max(), bool absoluteTolerance = false, bool useFrobeniusNorm = false, T* oldFrobeniusNorm = nullptr);


    //! small helper function to calculate the rank for truncating in the Frobenius norm with a given tolerance
    template<typename EigenSVD, typename T>
    int rankInFrobeniusNorm(const EigenSVD& svd, T absoluteTolerance)
    {
      constexpr auto pow2 = [](auto x){return x*x;};
      const T squaredTol = pow2(absoluteTolerance);
      int rank = svd.rank();
      T squaredError = 0;
      while( rank > 1 )
      {
        squaredError += pow2(svd.singularValues()(rank-1));
        if( squaredError > squaredTol )
          break;
        rank--;
      }
      return rank;
    }
  }


  //! split a fixed-size rank-3 tensor into 2 smaller tensors
  //!
  //! Split t3c into t3a and t3b such that
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)   with k=k2*na+k1 (respectively k=k1*nb+k2)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param[in]  t3c           rank-3 tensor
  //! @param[in]  na            second dimension (n) of the first result tensor t3a
  //! @param[in]  nb            second dimension (n) of the second result tensor t3b
  //! @param[in]  transpose     transpose second dimension (n): uses k = k1+k2*nb instead of k = k2*na+nb
  //! @param[in]  leftOrtog     make left part (t3a) orthogonal if true, otherwise t3b is made orthogonal
  //! @param[in]  rankTolerance truncation tolerance for determining the rank between t3a and t3b
  //! @param[in]  maxRank       maximal rank of the decomposition of t3c into t3a and t3b
  //! @return     [t3a, t3b]    the two parts of the splitted rank-3 tensor
  //!
  template<typename T>
  std::pair<Tensor3<T>,Tensor3<T>> split(const Tensor3<T>& t3c, int na, int nb, bool transpose, bool leftOrthog = true, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max());
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor3_split_impl.hpp"
#endif

#endif // PITTS_TENSOR3_SPLIT_HPP
