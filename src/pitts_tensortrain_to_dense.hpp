/*! @file pitts_tensortrain_to_dense.hpp
* @brief conversion from the tensor-train format into a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_TO_DENSE_HPP
#define PITTS_TENSORTRAIN_TO_DENSE_HPP

// includes
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_transform.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate fully dense tensor from a tensor-train decomposition (stored as a MultiVector)
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param TT         the tensor in tensor-train format
  //! @param X          multivector for storing the tensor in dense format (as one column)
  //!
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

    const int r0 = TT.subTensors().front().r1();
    const int rd = TT.subTensors().back().r2();
    const int nDim = TT.dimensions().size();

    const auto totalSize = r0 * rd * std::accumulate(begin(TT.dimensions()), end(TT.dimensions()), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());

    // use "multivector <- multivector * tensor2" ("transform") implementation
    // as in most cases the calculation becomes memory bound and requires lots of reshaping
    if( nDim == 1 )
    {
      const auto& subT = TT.subTensors()[0];
      X.resize(subT.r1() * subT.n() * subT.r2(), 1);
      for(int k = 0; k < subT.r2(); k++)
        for(int j = 0; j < subT.n(); j++)
          for(int i = 0; i < subT.r1(); i++)
            X(i+subT.r1()*j+subT.r1()*subT.n()*k,0) = subT(i,j,k);
    }
    else // nDim > 1
    {
      const auto& subT = TT.subTensors()[0];
      unfold_left(subT, X);
    }

    MultiVector<T> Y;
    Tensor2<T> M;
    for(int iDim = 1; iDim < nDim; iDim++)
    {
      // copy sub-tensor to Tensor2 to pass it to transform later
      const auto& subT = TT.subTensors()[iDim];
      unfold_right(subT, M);

      if( iDim+1 == nDim )
        transform(X, M, Y, {X.rows()*subT.n()*subT.r2(),1});
      else
        transform(X, M, Y, {X.rows()*subT.n(),subT.r2()});
      std::swap(X, Y);
    }
    assert(X.rows() == totalSize && X.cols() == 1);
  }


  //! calculate fully dense tensor from a tensor-train decomposition
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //! @tparam Iter      contiguous output iterator to write the dense data
  //!
  //! @param TT             the tensor in tensor-train format
  //! @param first          output iterator that points to the first index, e.g. std::begin(someContainer)
  //! @param last           output iterator that points behind the last index, e.g. std::end(someContainer)
  //!
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

    const int r0 = TT.subTensors().front().r1();
    const int rd = TT.subTensors().back().r2();

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


#endif // PITTS_TENSORTRAIN_TO_DENSE_HPP
