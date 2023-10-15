// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Manuel Joey Becklas
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_axpby.hpp
* @brief addition for simple tensor train format
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-10-26
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_HPP
#define PITTS_TENSORTRAIN_AXPBY_HPP

// includes
#include <limits>
#include "pitts_tensortrain.hpp"

namespace PITTS 
{

    //!
    //! @brief Scale and add one tensor train to another.
    //! 
    //! Calculate gamma * y <- alpha * x + beta * y, 
    //! such that the result y is orthogonalized and has frobenius norm 1.0
    //! 
    //! @tparam T underlying data type (double, complex, ...)
    //! 
    //! @param alpha         coefficient of tensor x, scalar value
    //! @param TTx           tensor x in tensor train format
    //! @param beta          coefficient of tensor y, scalar value
    //! @param TTy           tensor y in tensor train format, result tensor
    //! @param rankTolerance approximation accuracy that is used to reduce the TTranks of the result
    //! @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
    //! @return              norm of the result tensor
    //!
    template<typename T>
    T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());

} // namespace PITTS

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_axpby_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_AXPBY_HPP
