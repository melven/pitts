/*! @file pitts_tensortrain_axpby.hpp
* @brief addition for simple tensor train format
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-10-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensortrain_axpby.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSORTRAIN_AXPBY)
import pitts_tensortrain_axpby;
#define PITTS_TENSORTRAIN_AXPBY_HPP
#endif

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_HPP
#define PITTS_TENSORTRAIN_AXPBY_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#ifndef PITTS_USE_MODULES
#include "pitts_eigen.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#else
#include <string>
#include <complex>
#define EIGEN_CORE_MODULE_H
#include <Eigen/src/Core/util/Macros.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/ForwardDeclarations.h>
#endif
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby_plain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensortrain_axpby;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{

    /**
     * @brief Scale and add one tensor train to another.
     * 
     * Calculate gamma * y <- alpha * x + beta * y, 
     * such that the result y is orthogonalized and has frobenius norm 1.0
     * 
     * @tparam T underlying data type (double, complex, ...)
     * 
     * @param alpha         coefficient of tensor x, scalar value
     * @param TTx           tensor x in tensor train format
     * @param beta          coefficient of tensor y, scalar value
     * @param TTy           tensor y in tensor train format, result tensor
     * @param rankTolerance approximation accuracy that is used to reduce the TTranks of the result
     * @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
     * @return              norm of the result tensor
     */
    template<typename T>
    T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
    {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const std::vector<int>& x_dim = TTx.dimensions();
        const std::vector<int>& y_dim = TTy.dimensions();
        const int& d = x_dim.size(); // order d

        TT_Orthogonality x_ortho = TTx.isOrthogonal();
        TT_Orthogonality y_ortho = TTy.isOrthogonal();

        // check that dimensions match

        if (x_dim != y_dim)
            throw std::invalid_argument("TensorTrain axpby dimension mismatch!");
    
        if( TTx.subTensor(0).r1() != TTy.subTensor(0).r1() || TTx.subTensor(d-1).r2() != TTy.subTensor(d-1).r2() )
            throw std::invalid_argument("TensorTrain axpby boundary ranks mismatch!");

        // special cases
        
        if (std::abs(alpha) == 0 || std::abs(beta) == 0)
        {   
            // 0*TTx + 0*TTy = 0
            if (std::abs(alpha) == 0 && std::abs(beta) == 0)
                return 0;
            
            // if beta = 0, copy (alpha,TTx) -> (beta,TTy)
            if (std::abs(alpha) != 0)
            {
                copy(TTx, TTy); // TTx -> TTy
                y_ortho = x_ortho;
                beta = alpha;
            }
            
            // normalize and return alpha*TTy
            T sgn = (beta >= 0) ? 1.0 : -1.0;
            if (sgn < 0)
            {
                const int idx = (y_ortho == TT_Orthogonality::left) ? d - 1 : 0;
                TTy.editSubTensor(idx, [sgn](Tensor3<T>& subT){internal::t3_scale(sgn, subT);}, TTy.isOrthonormal(idx));
            }
            if (y_ortho == TT_Orthogonality::none)
                sgn *= leftNormalize(TTy, rankTolerance, maxRank);
            return beta * sgn;
        }

        // dispatch

        T gamma;

        if (x_ortho == TT_Orthogonality::none && y_ortho == TT_Orthogonality::none)
        {
            // no tensor is normalized
            gamma = internal::axpby_plain(alpha, TTx, beta, TTy, rankTolerance, maxRank);
        }
        else if (x_ortho != TT_Orthogonality::none && y_ortho != TT_Orthogonality::none)
        {
            // both tensors are normalized
            const int x_max_rank = *std::max_element(x_dim.begin(), x_dim.end());
            const int y_max_rank = *std::max_element(y_dim.begin(), y_dim.end());
            if (x_max_rank >= y_max_rank)
            {
                gamma = internal::axpby_normalized(alpha, TTx, beta, TTy, rankTolerance, maxRank);
            }
            else 
            {
                // goto y_ortho :/
                TensorTrain<T> TTtmp(TTx);
                gamma = internal::axpby_normalized(beta, TTy, alpha, TTtmp, rankTolerance, maxRank);
                std::swap(TTy, TTtmp);
            }
        }
        else if (x_ortho != TT_Orthogonality::none)
        {
            // only TTx is normalized
            gamma = internal::axpby_normalized(alpha, TTx, beta, TTy, rankTolerance, maxRank);
        }
        else //if (y_ortho != TT_Orthogonality::none)
        {
            // only TTy is normalized
            TensorTrain<T> TTtmp(TTx);
            gamma = internal::axpby_normalized(beta, TTy, alpha, TTtmp, rankTolerance, maxRank);
            std::swap(TTy, TTtmp);
        }

        return gamma;
    }

  // explicit template instantiations
  //template float axpby<float>(float alpha, const TensorTrain<float>& TTx, float beta, TensorTrain<float>& TTy, float rankTolerance, int maxRank);
  //template double axpby<double>(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double rankTolerance, int maxRank);
}

#endif // PITTS_TENSORTRAIN_AXPBY_HPP
