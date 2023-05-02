/*! @file pitts_tensortrain_axpby_normalized.hpp
* @brief addition for simple tensor train format where one of the tensors is normalized
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-09-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP
#define PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP

//#define VERBOSE

// includes
#include <cmath>
#include <limits>
#include "pitts_tensortrain.hpp"

namespace PITTS
{
    namespace internal
    {
        //! @brief Componentwise axpy for Tensor3 objects.
        //!
        //! y <- a*x + y
        //! 
        //! @tparam T    underlying data type
        //! @param a     scalar a
        //! @param x     [in] Tensor3 x
        //! @param y     [in,out] Tensor3 y
        //! 
        template<typename T>
        void t3_axpy(const T a, const Tensor3<T>& x, Tensor3<T>& y);

        //! @brief Returns if the tensor train A is (left or right)-orthogonal (up to some tolerance).
        //! 
        //! @tparam T        underlying type
        //! @param A         TensorTrain<T> object
        //! @param orthog    what type of orthogonality to check foe
        //! @return true     if A passes the orthogonality test
        //! @return false    if A fails the orthogonality test
        //! 
        template<typename T>
        bool is_normalized(const TensorTrain<T>& A, TT_Orthogonality orthog, double eps = 10*std::sqrt(std::numeric_limits<T>::epsilon()));

        //! @brief Left to right orthogonalization sweep for axpby_normalized function.
        //! This performs the axbpy operation as well as the orthogonalization.
        //! 
        //! @tparam T            underlying data type
        //! @param alpha         coefficient of tensor x, scalar value
        //! @param TTx_ortho     tensor x in tensor train format, left-orthogonal
        //! @param beta          coefficient of tensor y, scalar value
        //! @param TTy           tensor y in tensor train format
        //!
        template <typename T>
        void axpby_leftOrthogonalize(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy);

        
        //! @brief Right to left orthogonalization sweep for axpby_normalized function.
        //! This performs the axbpy operation as well as the orthogonalization.
        //! 
        //! @tparam T            underlying data type
        //! @param alpha         coefficient of tensor x, scalar value
        //! @param TTx_ortho     tensor x in tensor train format, right-orthogonal
        //! @param beta          coefficient of tensor y, scalar value
        //! @param TTy           tensor y in tensor train format
        //!
        template <typename T>
        void axpby_rightOrthogonalize(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy);
        

        //! @brief Add scaled tensor trains, where one of them (x) is normalized.
        //! 
        //! Calculate gamma * y <- alpha * x + beta * y, 
        //! such that the result y is orthogonalized and has frobenius norm 1.0
        //! 
        //! @warning Tensor x (TTx) must already be left- or right- orthogonal.
        //! @warning This function doesn't check that tensor dimensions match nor special cases. Call the function axpby for that.
        //! 
        //! @tparam T underlying data type (double, complex, ...)
        //! 
        //! @param alpha         coefficient of tensor x, scalar value
        //! @param TTx           orthogonalized tensor x in tensor train format
        //! @param beta          coefficient of tensor y, scalar value
        //! @param TTy           tensor y in tensor train format, result tensor
        //! @param rankTolerance approxiamtion accuracy that is used to reduce the TTranks of the result
        //! @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
        //! @return              norm of the result tensor
        template <typename T>
        T axpby_normalized(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());
    
    } // namespace internal

} // namespace PITTS

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_axpby_normalized_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP
