/*! @file pitts_tensortrain_to_dense_impl.hpp
* @brief conversion from the tensor-train format into a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_TO_DENSE_IMPL_HPP
#define PITTS_TENSORTRAIN_TO_DENSE_IMPL_HPP

// includes
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <cassert>
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_multivector_transform.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TT toDense (MultiVector variant)
  template<typename T>
  void toDense(const TensorTrain<T>& TT, MultiVector<T>& X)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    if( TT.dimensions().size() == 0 )
    {
      X.resize(0,1);
      return;
    }

    const int nDim = TT.dimensions().size();
    const int r0 = TT.subTensor(     0).r1();
    const int rd = TT.subTensor(nDim-1).r2();

    const auto totalSize = r0 * rd * std::accumulate(begin(TT.dimensions()), end(TT.dimensions()), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());

    // use "multivector <- multivector * tensor2" ("transform") implementation
    // as in most cases the calculation becomes memory bound and requires lots of reshaping
    const auto& subT0 = TT.subTensor(0);
    if( nDim == 1 )
    {
      unfold(subT0, X);
      return;
    }

    unfold_left(subT0, X);

    MultiVector<T> Y;
    for(int iDim = 1; iDim < nDim; iDim++)
    {
      // copy sub-tensor to Tensor2 to pass it to transform later
      const auto& subT = TT.subTensor(iDim);
      const auto& M = unfold_right(subT);

      const auto totalSize = X.rows()*subT.n()*subT.r2();
      const auto newCols = (iDim+1 == nDim) ? 1 : subT.r2();
      transform(X, M, Y, {totalSize/newCols, newCols});
      std::swap(X, Y);
    }
    assert(X.rows() == totalSize && X.cols() == 1);
  }


  // implement TT toDense (generic variant)
  template<typename T, class Iter>
  void toDense(const TensorTrain<T>& TT, Iter first, Iter last)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that the input is contiguous in memory
    //static_assert(std::is_base_of< std::contiguous_iterator_tag, typename std::iterator_traits<Iter>::iterator_category >::value, "toDense only works with contiguous iterators!");
    static_assert(std::is_base_of< std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category >::value, "toDense only works with contiguous iterators!");

    // abort early for zero dimensions
    if( TT.dimensions().size() == 0 )
    {
      if( last - first != 0 )
        throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::toDense");
      return;
    }

    const int nDim = TT.dimensions().size();
    const int r0 = TT.subTensor(     0).r1();
    const int rd = TT.subTensor(nDim-1).r2();

    const auto totalSize = r0 * rd * std::accumulate(begin(TT.dimensions()), end(TT.dimensions()), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    if( totalSize != last - first )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::toDense");

    // for now simply call function with MultiVector argument
    MultiVector<T> X;
    toDense(TT, X);

    // copy X to output
#pragma omp parallel for schedule(static)
    for(long long i = 0; i < totalSize; i++)
      first[i] = X(i,0);
  }

}


#endif // PITTS_TENSORTRAIN_TO_DENSE_IMPL_HPP
