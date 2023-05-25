/*! @file pitts_tensor2_eigen_adaptor.hpp
* @brief Get an Eigen::Map for a simple PITTS::Tensor2
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-31
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_EIGEN_ADAPTOR_HPP
#define PITTS_TENSOR2_EIGEN_ADAPTOR_HPP

// includes
#include "pitts_tensor2.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Get a const Eigen::Map for a Tensor2
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  auto ConstEigenMap(const ConstTensor2View<T>& t2)
  {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<const Matrix>(&t2(0,0), t2.r1(), t2.r2());
  }

  //! Get a mutable Eigen::Map for a Tensor2
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  auto EigenMap(Tensor2View<T> t2)
  {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<Matrix>(&t2(0,0), t2.r1(), t2.r2());
  }
}


#endif // PITTS_TENSOR2_EIGEN_ADAPTOR_HPP
