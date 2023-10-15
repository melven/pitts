// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_debug.hpp
* @brief Debug helper functions for PITT::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-14
*
* Currently provides operator overloads to calculate with tensor-trains.
* As this generates temporaries and uses sub-optimal kernels, this is only intended for e.g. assert-statements.
**/

// include guard
#ifndef PITTS_TENSORTRAIN_DEBUG_HPP
#define PITTS_TENSORTRAIN_DEBUG_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for debugging functionality
  namespace debug
  {
    // helpful operations for checking correctness of tensor-train algorithms
    using PITTS::norm2;
    using PITTS::dot;

    //! operator overload for TT scaling (alpha*TT)
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator*(T alpha, const TensorTrain<T>& ttX)
    {
        TensorTrain<T> result = ttX;
        if( ttX.dimensions().size() > 0 )
        {
            PITTS::Tensor3<T> newSubT0;
            copy(result.subTensor(0), newSubT0);
            internal::t3_scale(alpha, newSubT0);
            result.setSubTensor(0, std::move(newSubT0));
        }
        return result;
    }

    //! operator overload for TT scaling (TT*alpha)
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator*(const TensorTrain<T>& ttX, T alpha) {return alpha*ttX;}

    //! operator overload for TT addition (axpy with copy)
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator+(const TensorTrain<T>& ttX, const TensorTrain<T>& ttY)
    {
        TensorTrain<T> result = ttY;
        const T scale = axpby(T(1), ttX, T(1), result, T(0));
        return scale * result;
    }

    //! operator overload for TT subtraction (axpy with copy)
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator-(const TensorTrain<T>& ttX, const TensorTrain<T>& ttY)
    {
        TensorTrain<T> result = ttY;
        const T scale = axpby(T(1), ttX, T(-1), result, T(0));
        return scale * result;
    }

    //! operator overload for TT scaling (TT/alpha)
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator/(const TensorTrain<T>& ttX, T alpha) {return T(1)/alpha * ttX;}

  }
}

#endif // PITTS_TENSORTRAIN_DEBUG_HPP
