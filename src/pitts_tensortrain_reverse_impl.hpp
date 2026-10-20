// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_reverse_impl.hpp
* @brief reverse the order of the dimensions of a tensor train
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-10-20
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_REVERSE_IMPL_HPP
#define PITTS_TENSORTRAIN_REVERSE_IMPL_HPP

// includes
#include <utility>
#include <vector>
#include "pitts_tensortrain_reverse.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    // implement t3_transpose
    template<typename T>
    void t3_transpose(const Tensor3<T>& A, Tensor3<T>& B)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      const int r1 = A.r1();
      const int n = A.n();
      const int r2 = A.r2();
      B.resize(r2, n, r1);
      // small cores: a plain loop (writes contiguous in the first index of B)
      for(int k = 0; k < r1; k++)
        for(int i = 0; i < n; i++)
          for(int j = 0; j < r2; j++)
            B(j, i, k) = A(k, i, j);
    }
  }


  // implement TT reverse
  template<typename T>
  void reverse(const TensorTrain<T>& TTx, TensorTrain<T>& TTy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TTx.dimensions().size();
    std::vector<Tensor3<T>> cores(nDim);
    std::vector<TT_Orthogonality> orthonormal(nDim);
    for(int k = 0; k < nDim; k++)
    {
      internal::t3_transpose(TTx.subTensor(nDim-1-k), cores[k]);
      // left- and right-orthonormality swap (a core can be both)
      const TT_Orthogonality o = TTx.isOrthonormal(nDim-1-k);
      orthonormal[k] = ((o & TT_Orthogonality::left) != TT_Orthogonality::none ? TT_Orthogonality::right : TT_Orthogonality::none)
                     | ((o & TT_Orthogonality::right) != TT_Orthogonality::none ? TT_Orthogonality::left : TT_Orthogonality::none);
    }
    TTy = TensorTrain<T>(std::move(cores), orthonormal);
  }
}


#endif // PITTS_TENSORTRAIN_REVERSE_IMPL_HPP
