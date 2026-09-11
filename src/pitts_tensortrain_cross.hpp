// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross.hpp
* @brief TT-cross approximation: apply a function element-wise to a tensor in TT format
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Computes y ~ f(x) entry-wise for a tensor x given in TT format, without forming the dense tensor.
*
* Algorithm (cross interpolation, see Oseledets/Tyrtyshnikov 2010, Savostyanov/Oseledets 2011,
* Dolgov/Savostyanov 2020 and the amen_cross algorithm of the TT-Toolbox):
*
*   The result y is built only from entries of f(x) on "crosses", defined by nested index sets
*     I_k  = { (i_0,...,i_k) }        left multi-indices for bond k   (I_k subset of I_{k-1} x S_k),
*     J_k  = { (i_k,...,i_{d-1}) }    right multi-indices for bond k-1 (J_k subset of S_k x J_{k+1}),
*   with |I_k| = |J_{k+1}| = r_k (the TT rank of y at bond k) and I_{-1} = J_d = {()}.
*
*   y is kept in the interpolation ("CI-canonical") gauge: in a left-to-right sweep every processed core
*   Y_k restricted to the rows I_k of its left unfolding (r_{k-1} n_k x r_k) is the identity, in a
*   right-to-left sweep every processed core restricted to the columns J_k of its right unfolding is the
*   identity. Then y evaluated on the cross (I_{k-1}, S_k, J_{k+1}) is simply the core Y_k, and the
*   difference between the newly evaluated block f(x)(I_{k-1}, S_k, J_{k+1}) and Y_k is the current
*   approximation error on the cross (used for the convergence check).
*
*   One step of a left-to-right sweep at core k:
*     1. Evaluate the block B = f( x(I_{k-1}, S_k, J_{k+1}) ) of shape r_{k-1} x n_k x r_k.
*        The entries of x are obtained from cached interface matrices (partial contractions of the
*        cores of x on the nested index sets, see pitts_tensortrain_cross_helper.hpp), so the block is
*        one small contraction plus one call of f per entry.
*     2. Truncated SVD of the left unfolding of B (r_{k-1} n_k x r_k) with rankTolerance / maxRank.
*     3. Optional AMEn-style enrichment (nMALS = 1 only): append directions of the current error
*        z ~ f(x) - y (a second, small-rank cross approximation of the error, updated in the same sweep).
*     4. maxvol on the resulting orthogonal factor U selects the new nested index set I_k;
*        Y_k = U * U(I_k,:)^{-1} (interpolation gauge), the remainder is multiplied into Y_{k+1}.
*   The right-to-left sweep is the mirror image (right unfoldings, column selection, J_k) but without
*   step 3: it only truncates, so after every full sweep the ranks of y are the truncated ranks. The
*   enlarged left sets I_{k-1} (|I_{k-1}| = r_{k-1} + enrichmentRank) still allow the rank r_k to grow
*   in the right-to-left sweep (amen_cross enriches in both directions and needs a final truncation).
*   The error crosses (I^z, J^z) are updated in both directions.
*   With nMALS = 2 (DMRG-cross / TT-RC), step 1 evaluates the 2-core block
*   f( x(I_{k-1}, S_k, S_{k+1}, J_{k+2}) ) and step 2 splits it, which adapts the rank without an
*   error approximation but costs O(n^2) instead of O(n) entries per core.
*
*   Initialization: index sets from a maxvol sweep over the orthogonalized cores of x itself (no
*   function evaluations), initial y = f applied to the corresponding slices of x.
*
*   Cost per full sweep (nMALS = 1, ranks r of y, r_x of x, error rank rho, enrichment rank rho'):
*     function evaluations  sum_k 2 (r_{k-1} + rho') n_k r_k  (blocks)  +  r_{k-1} n_k rho + 2 rho n_k rho  (error crosses),
*     flops                 O(d n r r_x^2 + d n r^2 r_x) for the interface contractions and blocks,
*                           O(d n r^3) for SVD/maxvol.
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HPP
#define PITTS_TENSORTRAIN_CROSS_HPP

// includes
#include <cmath>
#include <limits>
#include <vector>
#include "pitts_tensortrain.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Options for the TT-cross approximation (applyElementwise)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  struct TTCrossOptions
  {
    //! relative truncation tolerance for the singular values of each local block (rank adaptation)
    //!
    //! Applied as rankTolerance/sqrt(d-1) per block like in solveMALS / dmrg_cross. Note that the blocks
    //! are truncated in the interpolation gauge, so unlike in solveMALS this is a heuristic and not a
    //! rigorous bound on the global error (the left/right parts of y are not orthonormal, only bounded).
    //!
    T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon());

    //! maximal TT rank of the result, enforced even if this violates rankTolerance
    int maxRank = std::numeric_limits<int>::max();

    //! maximal number of full sweeps (left-to-right + right-to-left)
    int nSweeps = 10;

    //! convergence: stop after a full sweep if max_k ||B_k - Y_k||_F / ||B_k||_F < convergenceTolerance
    //! (B_k: newly evaluated block, Y_k: previous core, both on the cross of core k). With enrichment the
    //! relative error ||f(x) - y||_F / ||B_k||_F on the error crosses of core k is included in the maximum
    //! (the change of y on its own crosses alone cannot detect a rank that is too low).
    T convergenceTolerance = std::sqrt(std::numeric_limits<T>::epsilon());

    //! number of cores treated together per step: 1 = ALS-cross with error-based enrichment,
    //! 2 = DMRG-cross (TT-RC), rank-adaptive through the SVD of the 2-core block
    int nMALS = 1;

    //! rank of the error approximation z ~ f(x) - y used for the enrichment (nMALS = 1 only, 0 = off)
    int errorRank = 3;

    //! number of error directions appended to each core per step (nMALS = 1 only, <= errorRank)
    int enrichmentRank = 2;

    //! initial index sets of the error approximation z: true = random right-nested sets (as in amen_cross),
    //! false = the first errorRank elements of the maxvol sets derived from x (nMALS = 1 only)
    bool randomErrorIndexSets = true;

    //! tolerance for the dominance criterion of the maxvol row selection (see internal::maxvol)
    T maxvolTolerance = T(0.05);

    //! print convergence information per sweep
    bool verbose = false;
  };


  //! Result information of a TT-cross run (applyElementwise)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  struct TTCrossInfo
  {
    //! number of full sweeps performed
    int nSweeps = 0;

    //! maximal relative change of a local block (and relative error on the error crosses) in the last sweep, see convergenceTolerance
    T lastChange = T(0);

    //! total number of evaluated tensor entries (calls of the element-wise function)
    long long nEvaluations = 0;

    //! true if convergenceTolerance was reached within nSweeps
    bool converged = false;

    //! final left index sets: leftIndices[k] holds the multi-indices (i_0,...,i_k) of I_k for k = 0..d-2, one per row
    //!
    //! y interpolates f(x) on the fibers (I_{k-1}, S_k, J_{k+1}) of the final crosses up to the truncation
    //! errors (exactly if nothing was truncated), so together with rightIndices these sets define the result
    //! (useful for tests and warm starts). The last sweep ends right-to-left, hence |J_{k+1}| = r_k is the
    //! final rank while I_k stems from the preceding left-to-right sweep and contains up to enrichmentRank
    //! additional multi-indices (nMALS = 1 with enrichment).
    //!
    std::vector<Eigen::MatrixXi> leftIndices;

    //! final right index sets: rightIndices[k] holds the multi-indices (i_k,...,i_{d-1}) of J_k for k = 1..d-1, one per row
    //! (these are the pivot sets of the final cores: |J_k| = r_{k-1}, and Y_k restricted to the columns J_k is the identity)
    std::vector<Eigen::MatrixXi> rightIndices;
  };


  //! Apply a function element-wise to a tensor in TT format using TT-cross approximation
  //!
  //! Computes TTy ~ f(TTx) entry-wise, i.e. y(i_0,...,i_{d-1}) = f(x(i_0,...,i_{d-1})),
  //! see the file documentation for the algorithm.
  //!
  //! @tparam T               underlying data type (double, complex, ...)
  //! @tparam Function        scalar functor T f(T), e.g. [](double v){return std::sqrt(v);}
  //!                         It is called once per evaluated entry inside the block loops, so a small
  //!                         inlinable functor is preferable to std::function.
  //!
  //! @param f                the element-wise function
  //! @param TTx              input tensor in TT format (boundary ranks r_0 = r_d = 1 required)
  //! @param TTy              result f(TTx) in TT format; the dimensions must match TTx, the input
  //!                         contents are ignored (index sets are initialized from TTx)
  //! @param options          algorithm parameters, see TTCrossOptions
  //! @return                 convergence information, see TTCrossInfo
  //!
  //! @throws std::invalid_argument for mismatching dimensions, boundary ranks != 1 or inconsistent options
  //!
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwise(const Function& f, const TensorTrain<T>& TTx, TensorTrain<T>& TTy,
                                  const TTCrossOptions<T>& options = TTCrossOptions<T>());
}

// applyElementwise is templated on the functor type, so it cannot be instantiated explicitly in a .cpp file:
// the implementation is always included (also for PITTS_DEVELOP_BUILD); only the non-templated helpers
// (pitts_tensortrain_cross_helper.hpp) are compiled separately in the development build.
#include "pitts_tensortrain_cross_impl.hpp"

#endif // PITTS_TENSORTRAIN_CROSS_HPP
