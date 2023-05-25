/*! @file pitts_tensor2_concat.hpp
* @brief Concatenize 2 tensors to one larger tensor (combine blocks of matrices)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-25
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_CONCAT_HPP
#define PITTS_TENSOR2_CONCAT_HPP

// includes
#include <optional>
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! horizontally concatenate matrices
  //!
  //! C = (A B)
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  //! @param A    desired left block of the result, set to zero if not provided, dimensions r1 x r2A
  //! @param B    desired right block of the result, set to zero if not provided, dimensions r1 x r2B
  //! @param C    result from concatenating A and B, dimensions must be r1 x (r2A+r2B)
  //!
  template<typename T>
  void concatLeftRight(const std::optional<ConstTensor2View<T>>& A, const std::optional<ConstTensor2View<T>>& B, Tensor2View<T> C);

  //! vertically concatenate matrices
  //!
  //! C = (A)
  //!     (B)
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  //! @param A    desired left block of the result, set to zero if not provided, dimensions r1A x r2
  //! @param B    desired right block of the result, set to zero if not provided, dimensions r1B x r2
  //! @param C    result from concatenating A and B, dimensions must be (r1A+r1B) x r2
  //!
  template<typename T>
  void concatTopBottom(const std::optional<ConstTensor2View<T>>& A, const std::optional<ConstTensor2View<T>>& B, Tensor2View<T> C);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_concat_impl.hpp"
#endif

#endif // PITTS_TENSOR2_CONCAT_HPP
