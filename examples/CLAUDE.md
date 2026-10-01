<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# examples/ — Python usage examples

Stand-alone scripts using the `pitts_py` module (run with `PYTHONPATH=build/src`). They are not part of the build or the tests. All follow the pattern `pitts_py.initialize()` ... `pitts_py.finalize()` and use the `*_double` classes. Several scripts re-implement Krylov methods in Python on top of the TT primitives (`apply`, `axpby`, `dot`, `normalize`) as algorithmic prototypes; the C++ library has its own `solveGMRES`/`solveMALS`.

| File | Content |
|---|---|
| `tt_laplace_operator.py` | Builds a d-dimensional Laplace operator as `TensorTrainOperator` (helper `LaplaceOperator(dims)`), reused by other scripts. |
| `tt_convection_operator.py` | Same for a convection (first-derivative) operator, `ConvectionOperator(dims)`. |
| `tt_solve_gmres.py` | Solves a Laplace system with the built-in `pitts_py.solveGMRES`. |
| `tt_solve_mals.py` | Solves a Laplace system with the built-in `pitts_py.solveMALS`. |
| `tt_gmres.py` | Pure-Python TT-GMRES (`tt_gmres`) with adaptive truncation tolerance. |
| `tt_fgmres.py` | Flexible TT-GMRES with variable preconditioning. |
| `tt_pivmgs.py` | Pivoted modified Gram-Schmidt for TT vectors (`tt_pivmgs`) with orthogonality checks. |
| `tt_gmres_pivmgs.py` | TT-GMRES using pivoted MGS orthogonalization. |
| `tt_gmres_pivmgs_smaller_eps.py` | Variant with tighter truncation tolerance. |
| `tt_gmres_pivmgs_leftprecond.py`, `tt_gmres_pivmgs_rightprecond.py` | Left- / right-preconditioned TT-GMRES variants. |
| `tt_rank1_preconditioner.py` | Rank-1 (Kronecker inverse) preconditioner class for TT operators. |
| `tt_ssor_preconditioner.py` | SSOR-type preconditioner in TT format using triangular Jacobi iterations. |
| `tt_arnoldi.py` | Arnoldi iteration in TT format (builds an orthonormal Krylov basis and the Hessenberg matrix). |
| `tt_jacobi_davidson.py` | Jacobi-Davidson eigenvalue solver prototype in TT format (`tt_jacobi_davidson`). |
| `tt_powers.py` | Repeated operator application (power iteration style), rank growth. |
| `axpby_bench.py` | Minimal timing of `pitts_py.axpby` on random TTs. |
| `tsqr_bench.py` | Compares PITTS TSQR runtime with `numpy.linalg.qr`. |
| `truncated_square_svd_approximation.py` | Uses TT-SVD to approximate the SVD of a large square low-rank matrix. |
| `spectral_clustering.py` | Spectral clustering example using a low-rank (TT) approximation of the data. |
| `big_data_tt.py` | Reads large data, reorders indices hierarchically and compresses it to TT format. |
| `analyze_tt.py` | Reads TT data stored as `.npz` dicts (from `big_data_tt.py`), rebuilds the TT and analyzes samples. |
| `numpy_to_tt_DeepArc.py` | Reads the DeepArc flow-field data (numpy arrays, video frames via OpenCV), converts it to TT format and reports runtime and memory usage per step. |
| `numpy_to_tt_DeepArc.log`, `numpy_to_tt_DeepArc_.log` | Output logs of two runs of `numpy_to_tt_DeepArc.py` on a cluster node (timings, memory, TT ranks). |
| `Interactive_DeepArc_examples.ipynb` | Jupyter notebook: interactive TT compression of the DeepArc data with video output (OpenCV/matplotlib) for comparing the original and the reconstruction. |
| `Interactive_DeepArc_thesis.ipynb` | Variant of the above with the plots and comparisons used in the thesis. |
