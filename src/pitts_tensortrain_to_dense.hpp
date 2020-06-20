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
#include <Eigen/Dense>
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format
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

    const auto totalSize = std::accumulate(begin(TT.dimensions()), end(TT.dimensions()), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    if( totalSize != last - first )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::toDense");

    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    EigenMatrix tmp = EigenMatrix::Identity(1,1);
    for(int iDim = 0; iDim < TT.dimensions().size(); iDim++)
    {
      // copy sub-tensor to matrix to make it easier
      const auto& subT = TT.subTensors()[iDim];
      EigenMatrix subT_matrix(subT.r1(), subT.n()*subT.r2());
      for(int k = 0; k < subT.r2(); k++)
        for(int j = 0; j < subT.n(); j++)
          for(int i = 0; i < subT.r1(); i++)
            subT_matrix(i, j + k*subT.n()) = subT(i,j,k);

      EigenMatrix newTmp = tmp * subT_matrix;
      tmp.resize(newTmp.rows()*subT.n(), newTmp.cols()/subT.n());
      tmp = Eigen::Map<EigenMatrix>(newTmp.data(), tmp.rows(), tmp.cols());
    }

    // copy to output
    Eigen::Map<EigenMatrix>(&(*first), totalSize, 1) = tmp;
  }

}


#endif // PITTS_TENSORTRAIN_TO_DENSE_HPP
