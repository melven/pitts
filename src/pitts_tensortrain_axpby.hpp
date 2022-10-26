/*! @file pitts_tensortrain_axpby.hpp
* @brief addition for simple tensor train format
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-10-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_HPP
#define PITTS_TENSORTRAIN_AXPBY_HPP

// includes
#include "pitts_tensortrain_axpby_plain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"

namespace PITTS 
{

    template<typename T>
    T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
    {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const std::vector<int>& x_dim = TTx.dimensions();
        const std::vector<int>& y_dim = TTy.dimensions();
        const int& d = x_dim.size(); // order d

        // check that dimensions match

        if (x_dim != y_dim)
            throw std::invalid_argument("TensorTrain axpby_normalized dimension mismatch!");
    
        if( TTx.subTensor(0).r1() != TTy.subTensor(0).r1() || TTx.subTensor(d-1).r2() != TTy.subTensor(d-1).r2() )
            throw std::invalid_argument("TensorTrain axpby_normalized boundary ranks not equal to 1!");

        // special cases

        if (std::abs(alpha) == 0)
            return beta;
        
        if (std::abs(beta) == 0)
        {
            copy(TTx, TTy); // TTx -> TTy
            return alpha;
        }

        // dispatch

        const TT_Orthogonality x_ortho = TTx.isOrthogonal();
        const TT_Orthogonality y_ortho = TTy.isOrthogonal();

        T gamma;

        if (x_ortho == TT_Orthogonality::none && y_ortho == TT_Orthogonality::none)
        {
            gamma = axpby_plain(alpha, TTx, beta, TTy, rankTolerance, maxRank);
        }
        else if (x_ortho != TT_Orthogonality::none && y_ortho != TT_Orthogonality::none)
        {
            const int x_max_rank = *std::max_element(x_dim.begin(), x_dim.end());
            const int y_max_rank = *std::max_element(y_dim.begin(), y_dim.end());
            if (x_max_rank >= y_max_rank)
            {
                gamma = axpby_normalized(alpha, TTx, beta, TTy, rankTolerance, maxRank);
            }
            else 
            {
                // goto y_ortho :/
                TensorTrain<T> TTtmp(TTx);
                gamma = axpby_normalized(beta, TTy, alpha, TTtmp, rankTolerance, maxRank);
                std::swap(TTy, TTtmp);
            }
        }
        else if (x_ortho != TT_Orthogonality::none)
        {
            gamma = axpby_normalized(alpha, TTx, beta, TTy, rankTolerance, maxRank);
        }
        else if (y_ortho != TT_Orthogonality::none)
        {
            TensorTrain<T> TTtmp(TTx);
            gamma = axpby_normalized(beta, TTy, alpha, TTtmp, rankTolerance, maxRank);
            std::swap(TTy, TTtmp);
        }

        return gamma;
    }

} // namespace PITTS

#endif // PITTS_TENSORTRAIN_AXPBY_HPP