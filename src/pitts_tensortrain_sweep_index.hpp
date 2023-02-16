/*! @file pitts_tensortrain_sweep_index.hpp
* @brief Helper class for MALS type tensor-train algorithms
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensortrain_sweep_index.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSORTRAIN_SWEEP_INDEX)
import pitts_tensortrain_sweep_index;
#define PITTS_TENSORTRAIN_SWEEP_INDEX_HPP
#endif

// include guard
#ifndef PITTS_TENSORTRAIN_SWEEP_INDEX_HPP
#define PITTS_TENSORTRAIN_SWEEP_INDEX_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <algorithm>
#include <stdexcept>
#include <vector>

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensortrain_sweep_index;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Small helper class for sweeping from left to right with different overlap and step sizes
    class SweepIndex final
    {
      public:
        //! setup a new SweepIndex starting at Index 0
        //!
        //! @param nMALS              number of sub-tensors to combine as one local problem (1 for ALS, 2 for MALS, nDim for global GMRES)
        //! @param nOverlap           overlap (number of sub-tensors) of two consecutive local problems in one sweep (0 for ALS 1 for MALS, must be < nMALS)
        //!
        SweepIndex(int nDim, int nMALS, int nOverlap, int initialLeftDim = 0) : nDim_(nDim), nMALS_(nMALS), nOverlap_(nOverlap), leftDim_(initialLeftDim)
        {
          if( nMALS_ < 1 || nMALS_ > nDim_ )
            throw std::invalid_argument("Tensortrain MALS SweepIndex: invalid parameter nMALS (1 <= nMALS)!");
          if( nOverlap_ < 0 || nOverlap_ >= nMALS_ )
            throw std::invalid_argument("Tensortrain MALS SweepIndex: invalid parameter nOverlap (1 <= nOverlap < nMALS)!");
        }

        //! check if the Sweep Index is valid
        [[nodiscard]] explicit operator bool() const noexcept
        {
          return leftDim() >= 0 && rightDim() < nDim_;
        }

        //! get number of dimensions
        int nDim() const {return nDim_;}

        //! get number of sub-tensors to combine for the local problem
        int nMALS() const {return nMALS_;}

        //! get overlap (num ber of sub-tensors) of two consecutive  steps
        int nOverlap() const {return nOverlap_;}

        //! left-most sub-tensor of current sub-problem
        int leftDim() const {return leftDim_;}

        //! right-most sub-tensor of current sub-problem
        int rightDim() const {return leftDim_ + nMALS_ - 1;}

        //! get the next SweepIndex (one step to the right)
        //!
        //! returns an invalid index if already completely right...
        //!
        SweepIndex next() const
        {
          if( rightDim() == nDim_-1 )
            return SweepIndex(nDim_, nMALS_, nOverlap_, nDim_);
          return SweepIndex(nDim_, nMALS_, nOverlap_, std::min(nDim_ - nMALS_, leftDim_ + nMALS_ - nOverlap_));
        }

        //! get previous SweepIndex (one step to the left)
        //!
        //! returns an invalid index if already completely left...
        //!
        SweepIndex previous() const
        {
          if( leftDim() == 0 )
            return SweepIndex(nDim_, nMALS_, nOverlap_, -1);
          return SweepIndex(nDim_, nMALS_, nOverlap_, std::max(0, leftDim_ - nMALS_ + nOverlap_));
        }

        //! get the first SweepIndex (left-most step)
        SweepIndex first() const {return SweepIndex(nDim_, nMALS_, nOverlap_, 0);};

        //! get the first SweepIndex (right-most step)
        SweepIndex last() const {return SweepIndex(nDim_, nMALS_, nOverlap_, nDim_ - nMALS_);};

        //! allow comparison of indices
        bool operator==(const SweepIndex&) const = default;

      private:
        //! number of dimensions
        int nDim_;

        //! number of sub-tensors to combine for the local problem
        int nMALS_;

        //! overlap (num ber of sub-tensors) of two consecutive  steps
        int nOverlap_;

        //! index of left-most sub-tensor of current sub-problem
        int leftDim_;
    };
  }
}


#endif // PITTS_TENSORTRAIN_SWEEP_INDEX_HPP
