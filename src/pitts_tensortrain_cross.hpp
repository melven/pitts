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
* Algorithm (cross interpolation, see [1]-[4] below and the amen_cross algorithm of the TT-Toolbox):
*
*   The result y is built only from entries of f(x) on "crosses", defined by nested index sets
*     I_k  = { (i_0,...,i_k) }        left multi-indices for bond k   (I_k subset of I_{k-1} x S_k),
*     J_k  = { (i_k,...,i_{d-1}) }    right multi-indices for bond k-1 (J_k subset of S_k x J_{k+1}),
*   with I_{-1} = J_d = {()}. The TT rank of y at bond k is r_k = |J_{k+1}|; |I_k| = r_k as well, except
*   directly after a left-to-right sweep with enrichment, where I_k holds up to enrichmentRank more
*   multi-indices than the current rank (see TTCrossInfo::leftIndices).
*
*   y is kept in the interpolation ("CI-canonical" in [4]) gauge: in a left-to-right sweep every processed
*   core Y_k restricted to the rows I_k of its left unfolding (|I_{k-1}| n_k x r_k) is the identity, in a
*   right-to-left sweep every processed core restricted to the columns J_k of its right unfolding is the
*   identity. Then y evaluated on the cross (I_{k-1}, S_k, J_{k+1}) is simply the core Y_k, and the
*   difference between the newly evaluated block f(x)(I_{k-1}, S_k, J_{k+1}) and Y_k is the current
*   approximation error on the cross (used for the convergence check).
*
*   One step of a left-to-right sweep at core k:
*     1. Evaluate the block B = f( x(I_{k-1}, S_k, J_{k+1}) ) of shape |I_{k-1}| x n_k x |J_{k+1}|.
*        The entries of x come from cached partial contractions of the cores of x on the nested index
*        sets (see pitts_tensortrain_cross_helper.hpp), so the block costs one small contraction plus
*        one call of f per entry (or a single call of f for the whole block for an array function).
*     2. Truncated SVD of the left unfolding of B (|I_{k-1}| n_k x |J_{k+1}|) with rankTolerance / maxRank.
*     3. Optional AMEn-style enrichment (nMALS = 1 only): append directions of the current error
*        z ~ f(x) - y (a second, small-rank cross approximation of the error, updated in the same sweep).
*     4. maxvol on the resulting orthogonal factor U selects the new nested index set I_k;
*        Y_k = U * U(I_k,:)^{-1} (interpolation gauge), the remainder is multiplied into Y_{k+1}.
*   The right-to-left sweep is the mirror image (right unfoldings, column selection, J_k) but without
*   step 3: it only truncates, so after every full sweep the ranks of y are the truncated ranks. The
*   enlarged left sets I_{k-1} of the preceding left-to-right sweep still allow the rank r_k to grow in
*   the right-to-left sweep (amen_cross enriches in both directions and needs a final truncation).
*   The error crosses (I^z, J^z) are updated in both directions.
*   With nMALS = 2 (DMRG-cross / TT-RC [2]), step 1 evaluates the 2-core block
*   f( x(I_{k-1}, S_k, S_{k+1}, J_{k+2}) ) and step 2 splits it, which adapts the rank without an
*   error approximation but costs O(n^2) instead of O(n) entries per core.
*
*   Initialization: index sets from a maxvol sweep over the orthogonalized cores of x itself (no
*   function evaluations), initial y = f applied to the corresponding slices of x.
*
*   Cost per full sweep (nMALS = 1, ranks r of y, r_x of x, error rank rho, enrichment rank rho'):
*     function evaluations  sum_k 2 (r_{k-1} + rho') n_k r_k  (blocks)
*                           + sum_k (r_{k-1} + rho') n_k rho + 2 rho n_k rho  (error crosses),
*     flops                 O(d n r r_x^2 + d n r^2 r_x) for the partial contractions and blocks,
*                           O(d n r^3) for SVD/maxvol.
*
* References:
*   [1] Oseledets, Tyrtyshnikov: "TT-cross approximation for multidimensional arrays",
*       Linear Algebra Appl. 432 (1) (2010) 70-88, doi 10.1016/j.laa.2009.07.024
*       -- nested index sets, cross interpolation of a tensor train, maxvol pivoting, and the
*          orthogonalization before the maxvol sweep (used for the initial index sets here).
*   [2] Savostyanov, Oseledets: "Fast adaptive interpolation of multi-dimensional arrays in tensor train
*       format", Proc. 7th International Workshop on Multidimensional Systems (nDS), IEEE, 2011,
*       doi 10.1109/nDS.2011.6076873
*       -- DMRG-cross / TT-RC: the two-core block with an SVD in between, the nMALS = 2 variant here.
*   [3] Dolgov, Savostyanov: "Parallel cross interpolation for high-precision calculation of
*       high-dimensional integrals", Comput. Phys. Commun. 246 (2020) 106869,
*       doi 10.1016/j.cpc.2019.106869
*       -- the sweep formulation used here, and the remark that a nested index set becomes invalid when
*          the neighboring set it refers to is replaced (Remark 5, see NestedIndices).
*   [4] Nunez Fernandez et al.: "Learning tensor networks with tensor cross interpolation: new algorithms
*       and libraries", SciPost Phys. 18, 104 (2025), doi 10.21468/SciPostPhys.18.3.104 (arXiv:2407.02454)
*       -- review of tensor cross interpolation: the CI-canonical (interpolation) gauge and the
*          classification into 1-site and 2-site sweeps with their pivot handling.
*   [5] Ritter et al.: "Quantics Tensor Cross Interpolation for High-Resolution Parsimonious
*       Representations of Multivariate Functions", Phys. Rev. Lett. 132, 056501 (2024),
*       doi 10.1103/PhysRevLett.132.056501 (arXiv:2303.11819)
*       -- the quantics (QTT) setting with many small dimensions, where rank growth during the sweeps
*          matters most (see the qtt test case).
*   [6] Goreinov, Oseledets, Savostyanov, Tyrtyshnikov, Zamarashkin: "How to find a good submatrix", in:
*       Matrix Methods: Theory, Algorithms, Applications, World Scientific, 2010, pp. 247-256,
*       doi 10.1142/9789812836021_0015
*       -- the maxvol algorithm, see pitts_tensor2_maxvol.hpp.
*   The AMEn-style enrichment by a separate low-rank approximation of the error follows the amen_cross
*   algorithm of the TT-Toolbox and the subspace enrichment of PITTS::solveMALS (nAMEnEnrichment).
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
    //! Applied as rankTolerance/sqrt(d-1) per block, like the truncation in leftNormalize/rightNormalize.
    //! Note that the blocks are truncated in the interpolation gauge, so unlike for an orthogonalized TT
    //! this is a heuristic and not a rigorous bound on the global error (the left/right parts of y are
    //! not orthonormal, their entries are only bounded by the maxvol dominance criterion).
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
    //! false = the maxvol sets of a rank-errorRank truncation of x (nMALS = 1 only)
    bool randomErrorIndexSets = true;

    //! tolerance for the dominance criterion of the maxvol row/column selection (see internal::maxvol)
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
    //! (these are the pivot sets of the final cores: |J_k| = r_{k-1}, and Y_k restricted to the columns J_k is the
    //! identity); the vector has d entries, rightIndices[0] is empty because J_0 does not exist
    std::vector<Eigen::MatrixXi> rightIndices;
  };


  //! Apply a function element-wise to a tensor in TT format using TT-cross approximation
  //!
  //! Computes TTy ~ f(TTx) entry-wise, i.e. y(i_0,...,i_{d-1}) = f(x(i_0,...,i_{d-1})),
  //! see the file documentation for the algorithm.
  //!
  //! @tparam T               underlying data type (double, complex, ...)
  //! @tparam Function        scalar functor T f(T), e.g. [](double v){return std::sqrt(v);}
  //!                         It is called once per evaluated entry inside the block loops (in parallel, so it
  //!                         must be pure), thus a small inlinable functor is preferable to std::function.
  //!                         Alternatively an array functor f(MultiVector<T>&) transforming all entries of a
  //!                         block in-place (a single column of r1*n*r2 rows, dimensions must be kept): it is
  //!                         called once per block and handles its own parallelization. Use this for functions
  //!                         with a significant per-call overhead, e.g. a numpy function called from Python.
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
