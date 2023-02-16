/*! @file pitts_tensortrain_debug.hpp
* @brief Debug helper functions for PITT::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-14
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
* Currently provides operator overloads to calculate with tensor-trains.
* As this generates temporaries and uses sub-optimal kernels, this is only intended for e.g. assert-statements.
**/

// just import the module if we are in module mode and this file is not included from pitts_tensortrain_debug.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSORTRAIN_DEBUG)
import pitts_tensortrain_debug;
#define PITTS_TENSORTRAIN_DEBUG_HPP
#endif

// include guard
#ifndef PITTS_TENSORTRAIN_DEBUG_HPP
#define PITTS_TENSORTRAIN_DEBUG_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_normalize.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensortrain_debug;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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

  // explicit template instantiations
}

#endif // PITTS_TENSORTRAIN_DEBUG_HPP
