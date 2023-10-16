// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Manuel Joey Becklas
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_axpby_impl.hpp
* @brief addition for simple tensor train format
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-10-26
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_IMPL_HPP
#define PITTS_TENSORTRAIN_AXPBY_IMPL_HPP

// includes
#include <algorithm>
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensortrain_axpby_plain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_performance.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensortrain_normalize.hpp"

namespace PITTS 
{

    // implement TT axpby
    template<typename T>
    T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance, int maxRank)
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
        if (d == 1)
        {
          const auto axpy_1d = [&](Tensor3<T>& subTy)
          {
            internal::t3_scale(beta, subTy);
            internal::t3_axpy(alpha, TTx.subTensor(0), subTy);
          };
          TTy.editSubTensor(0, axpy_1d);
          return leftNormalize(TTy, rankTolerance, maxRank);
        }

        // dispatch
#ifdef PITTS_TENSORTRAIN_PLAIN_AXPBY
        return internal::axpby_plain(alpha, TTx, beta, TTy, rankTolerance, maxRank);
#endif

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

} // namespace PITTS

#endif // PITTS_TENSORTRAIN_AXPBY_IMPL_HPP
