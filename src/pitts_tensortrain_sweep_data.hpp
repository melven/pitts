// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_sweep_data.hpp
* @brief Helper class for caching partial contractions in sweeping algorithms (MALS, TT-cross)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-05
*
* Moved out of pitts_tensortrain_solve_mals_helper.hpp so that other sweeping algorithms can reuse it.
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SWEEP_DATA_HPP
#define PITTS_TENSORTRAIN_SWEEP_DATA_HPP

// includes
#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! optional const reference (e.g. the previous partial contraction, which does not exist at the boundary)
    template<typename T>
    using optional_cref = std::optional<std::reference_wrapper<const T>>;

    //! Cache for quantities of a sweeping algorithm that are computed from the left and from the right
    //!
    //! Stores one ResultType per dimension. result_[iDim] is valid if iDim <= leftDim (computed from the left)
    //! or iDim >= rightDim (computed from the right). The user provides two functions that compute the
    //! result of one dimension from the previous one (leftToRight(iDim, prev, result) with prev the result
    //! of iDim-1, or nullopt at the boundary; rightToLeft analogously from iDim+1).
    //!
    template<typename ResultType, typename LeftToRightFunction, typename RightToLeftFunction>
    class SweepData final
    {
      public:
        SweepData(int nDim, LeftToRightFunction&& leftToRight, RightToLeftFunction rightToLeft) :
          nDim_(nDim), leftDim_(-1), rightDim_(nDim_), result_(nDim),
          updateLeftToRight_(std::move(leftToRight)),
          updateRightToLeft_(std::move(rightToLeft))
        {
        }
        [[nodiscard]] int nDim() const {return nDim_;}
        [[nodiscard]] int leftDim() const {return leftDim_;}
        [[nodiscard]] int rightDim() const {return rightDim_;}
        [[nodiscard]] const ResultType& subTensor(int iDim) const
        {
          assert((iDim >= 0 && iDim <= leftDim_) || (iDim >= rightDim_ && iDim <= nDim_-1));
          return result_[iDim];
        }
        [[nodiscard]] const std::vector<ResultType>& data() const {return result_;}
        //! mark the results of iDim..iDim+nMALS-1 (and everything depending on them) as invalid
        void invalidate(int iDim, int nMALS = 1)
        {
          iDim = std::clamp(iDim, 0, nDim_-1);
          leftDim_ = std::min(leftDim_, iDim-1);
          rightDim_ = std::max(rightDim_, iDim+nMALS);
        }
        //! compute all results up to newLeftDim from the left and from newRightDim from the right
        void update(int newLeftDim, int newRightDim)
        {
          assert(leftDim_ >= -1 && leftDim_ < rightDim_ && rightDim_ <= nDim_);
          assert(newLeftDim >= -1 && newLeftDim < newRightDim && newRightDim <= nDim_);

          for(; leftDim_ < newLeftDim; leftDim_++)
            updateLeftToRight_(leftDim_+1, left(), result_[leftDim_+1]);
          leftDim_ = newLeftDim;

          for(; rightDim_ > newRightDim; rightDim_--)
            updateRightToLeft_(rightDim_-1, right(), result_[rightDim_-1]);
          rightDim_ = newRightDim;
        }
        //! last valid result from the left (nullopt if none)
        [[nodiscard]] optional_cref<ResultType> left() const
        {
          if(leftDim_ < 0)
            return std::nullopt;
          return std::ref(result_[leftDim_]);
        }
        //! first valid result from the right (nullopt if none)
        [[nodiscard]] optional_cref<ResultType> right() const
        {
          if(rightDim_ >= nDim_)
            return std::nullopt;
          return std::cref(result_[rightDim_]);
        }
      private:
        int nDim_;
        int leftDim_ = 0;
        int rightDim_ = nDim_-1;
        std::vector<ResultType> result_;
        LeftToRightFunction updateLeftToRight_;
        RightToLeftFunction updateRightToLeft_;
    };

    //! for template argument deduction...
    template<typename ResultType, typename LeftToRightFunction, typename RightToLeftFunction>
    SweepData<ResultType, LeftToRightFunction, RightToLeftFunction> defineSweepData(int nDim, LeftToRightFunction&& leftToRight, RightToLeftFunction rightToLeft)
    {
      return {nDim, std::forward<LeftToRightFunction>(leftToRight), std::forward<RightToLeftFunction>(rightToLeft)};
    }
  }
}


#endif // PITTS_TENSORTRAIN_SWEEP_DATA_HPP
