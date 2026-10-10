// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross.hpp
* @brief TT-cross approximation: apply a function element-wise to one or two tensors in TT format
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Computes y ~ f(x) entry-wise for a tensor x given in TT format, without forming the dense tensor.
* Also for a function of two tensors, y ~ f(x1, x2), e.g. the element-wise product: then every block of
* x below is a pair of blocks of x1 and x2 on the same cross, and f combines them entry by entry.
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
*        z ~ f(x) - y (a second, small-rank cross approximation of the error, updated in the same sweep):
*        the error of the block is projected onto the complement of U, then its dominant (at most enrichmentRank)
*        directions are appended (as in PITTS::solveMALS).
*     4. maxvol on the resulting orthogonal factor U selects the new nested index set I_k;
*        Y_k = U * U(I_k,:)^{-1} (interpolation gauge), the remainder is multiplied into Y_{k+1}.
*     (With oversampling > 0, step 4 selects more rows than directions (rectangular maxvol [9]) and
*      Y_k = U * U(I_k,:)^+ with the pseudo-inverse of the tall matrix U(I_k,:): the cross is
*      interpolated in the least-squares sense instead of exactly [8]. The interpolation error of a square
*      cross exceeds the best low-rank error by a rank dependent factor [7], which can make the error grow with
*      the rank when the ranks are too low. The identity property of the gauge is then lost, y on its cross is
*      obtained by contracting the interfaces of y, and the TT ranks of y during the sweeps are the sizes of the
*      index sets; the final y is truncated.)
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
*   function evaluations; for two inputs on their stacked right unfoldings), and a zero y with these ranks that
*   the first left-to-right sweep replaces core by core.
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
*   [7] Savostyanov: "Quasioptimality of maximum-volume cross interpolation of tensors", Linear Algebra Appl.
*       458 (2014) 217-244 (arXiv:1305.1818)
*       -- the accuracy of the maximum-volume cross compared to the best approximation (factors (r+1)^2 per
*          bond in the Chebyshev norm, compounding over the dimension tree).
*   [8] Qin, Lidiak, Gong, Tang, Wakin, Zhu: "Error Analysis of Tensor-Train Cross Approximation", Advances in
*       Neural Information Processing Systems 35 (NeurIPS 2022), arXiv:2207.04327
*       -- error bound in the Frobenius norm for any selection of rows and columns, in terms of the
*          pseudo-inverses of the pivot matrices; with more rows/columns than the rank and a truncated
*          pseudo-inverse the cross approximation stays stable (basis of the oversampling option).
*   [9] Mikhalev, Oseledets: "Rectangular maximum-volume submatrices and their applications", Linear Algebra
*       Appl. 538 (2018) 187-211, arXiv:1502.07838
*       -- maximum-volume selection of more rows than columns, see internal::maxvol (nRows).
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
    //! relative truncation tolerance for each local block in the Frobenius norm (rank adaptation)
    //!
    //! Applied as rankTolerance/sqrt(d-1) per block, like the truncation in leftNormalize/rightNormalize.
    //! Note that the blocks are truncated in the interpolation gauge, so unlike for an orthogonalized TT
    //! this is a heuristic and not a rigorous bound on the global error (the left/right parts of y are
    //! not orthonormal, their entries are only bounded by the maxvol dominance criterion).
    //!
    T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon());

    //! maximal TT rank of the result, enforced even if this violates rankTolerance
    int maxRank = std::numeric_limits<int>::max();

    //! maximal TT rank per bond (d-1 entries, bond k between cores k and k+1); if given, it replaces maxRank
    std::vector<int> maxRanks = {};

    //! maximal number of full sweeps (left-to-right + right-to-left)
    int nSweeps = 10;

    //! convergence: stop after a full sweep if both measures of TTCrossInfo (change of y on its own crosses
    //! and, with enrichment, the error on the error crosses) are below this tolerance and no TT rank grew in
    //! that sweep (a growing rank means the algorithm itself is not done yet)
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

    //! oversampling of the interpolation sets relative to the local rank r (0 = none)
    //!
    //! Without oversampling a bond has exactly r rows/columns and y interpolates f(x) exactly on the cross
    //! (collocation). Compared to the best rank-r approximation, the error of such a cross is amplified by a
    //! factor that grows with r (up to about sqrt(r m) per bond for the maximum-volume choice) and compounds
    //! over the bonds. For ranks that are still far below the numerical rank of f(x) this can make the error
    //! larger with every enrichment step, and the estimated error as well as the sampled error then grow with
    //! the rank. With oversampling > 0 the rows (columns) are chosen by the oversampled maxvol,
    //! |I_k| = r + ceil(oversampling r), and the cores are the least-squares interpolation with the
    //! pseudo-inverse of the tall pivot matrix (cf. Qin et al., NeurIPS 2022): y only fits f(x) on the cross
    //! in the least-squares sense, and the amplification is much smaller (the norm of the pseudo-inverse
    //! decreases with the number of rows). In the sweeps the TT ranks
    //! of y are the sizes of the sets |I_k|, |J_k| (the extra directions carry no information; the rank itself only
    //! grows by the enrichment as without oversampling, nMALS = 1), after the last
    //! sweep y is truncated with rankTolerance and maxRank. The blocks that are evaluated grow with the
    //! sets in both dimensions, so the number of function evaluations and the cost per sweep grow by up to
    //! (1 + oversampling)^2. Typical values are 0.1 to 1.
    //!
    T oversampling = T(0);

    //! number of randomly drawn tensor entries used as an independent error check (0 = off)
    //!
    //! Costs this many additional function evaluations (once, the entries are fixed) and an evaluation of y
    //! at these entries after every half sweep (also printed with verbose). The entries are drawn once, so the
    //! resulting TTCrossInfo::lastSampleError is comparable between sweeps. Unlike the other two measures, which only see the crosses the algorithm itself selected,
    //! this is an unbiased estimate of the global error - it does not enter the convergence check,
    //! because its statistical error (roughly 1/sqrt(2*controlSampleSize)) is usually larger than
    //! convergenceTolerance.
    //!
    int controlSampleSize = 0;

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

    //! maximal change of y on its own crosses in the last sweep: max_k rms(B_k - Y_k) / rmsRef
    //!
    //! B_k is the newly evaluated block and Y_k the current core on the cross of core k (both are samples
    //! of f(x) resp. y at the cross indices), rms(.) = ||.||_F / sqrt(number of entries) and rmsRef the
    //! largest rms(B_k) seen so far. The rms scaling makes blocks of different size comparable (the crosses
    //! grow with the TT ranks, the error crosses have a different fixed size), the global reference prevents
    //! a bond whose cross only samples small values from dominating the maximum.
    //!
    //! Note that this is an average over the cross entries and that the crosses change between sweeps (new
    //! maxvol pivots are picked where the approximation is worst), so it is an indicator, not a monotone
    //! error bound: it typically increases in the sweeps where the ranks grow.
    //!
    T lastChange = T(0);

    //! maximal error estimate on the error crosses in the last sweep: max_k rms(f(x) - y) / rmsRef (0 without enrichment)
    //!
    //! Measured on the (independent) crosses of the error approximation z, so unlike lastChange this also
    //! detects a rank that is too low: y can be exact on its own crosses and still be a bad approximation.
    //! Still only an indicator: a rank-errorRank cross cannot see an error outside the fibers it samples.
    //!
    T lastError = T(0);

    //! relative error ||f(x) - y||_F / ||f(x)||_F estimated on controlSampleSize random entries (0 if off)
    //!
    //! The only measure that does not depend on the crosses chosen by the algorithm, hence unbiased - but
    //! with a statistical error of roughly 1/sqrt(2*controlSampleSize), see TTCrossOptions.
    //!
    T lastSampleError = T(0);

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


  //! Apply a function element-wise to two tensors in TT format using TT-cross approximation
  //!
  //! Computes TTy ~ f(TTx1, TTx2) entry-wise, i.e. y(i_0,...,i_{d-1}) = f(x1(i_0,...,i_{d-1}), x2(i_0,...,i_{d-1})),
  //! e.g. the element-wise (Hadamard) product with f(a,b) = a*b. Same algorithm and options as for a single input
  //! (see the file documentation): the inputs only enter through their entries on the crosses of y, so their exact
  //! combination (for the Hadamard product a tensor train with the ranks r^(1) r^(2)) is never formed; the cost
  //! of the contractions grows linearly with the number of inputs.
  //!
  //! The initial crosses are selected from the joint right parts of both inputs (initial rank at most
  //! r^(1) + r^(2)). If the result needs a much larger rank, it grows by at most enrichmentRank per sweep
  //! (nMALS = 1), so increase enrichmentRank / nSweeps or use nMALS = 2 in that case.
  //!
  //! @tparam T               underlying data type (double, complex, ...)
  //! @tparam Function        scalar functor T f(T, T), e.g. [](double a, double b){return a*b;}, or array functor
  //!                         f(MultiVector<T>& a, const MultiVector<T>& b) that stores the result in a, see the
  //!                         single-input variant
  //!
  //! @param f                the element-wise function
  //! @param TTx1             first input tensor in TT format (boundary ranks 1)
  //! @param TTx2             second input tensor in TT format (same dimensions, boundary ranks 1, ranks may differ)
  //! @param TTy              result f(TTx1, TTx2) in TT format, see the single-input variant
  //! @param options          algorithm parameters, see TTCrossOptions
  //! @return                 convergence information, see TTCrossInfo
  //!
  //! @throws std::invalid_argument for mismatching dimensions, boundary ranks != 1 or inconsistent options
  //!
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwise(const Function& f, const TensorTrain<T>& TTx1, const TensorTrain<T>& TTx2, TensorTrain<T>& TTy,
                                  const TTCrossOptions<T>& options = TTCrossOptions<T>());
}

// applyElementwise is templated on the functor type, so it cannot be instantiated explicitly in a .cpp file:
// the implementation is always included (also for PITTS_DEVELOP_BUILD); only the non-templated helpers
// (pitts_tensortrain_cross_helper.hpp) are compiled separately in the development build.
#include "pitts_tensortrain_cross_impl.hpp"

#endif // PITTS_TENSORTRAIN_CROSS_HPP
