<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# bench/ — benchmark scripts and archived results

High-level overview only. This directory is ~2.4 GB, mostly SLURM output logs (`*.sh-<jobid>.out`, `*.log`, up to ~60 MB each). **Do not read, grep or glob these logs wholesale**; they are archived measurement data, not code. Nothing here is part of the CMake build.

## Conventions

- `<name>.sh`: SLURM batch script (`#SBATCH` header, then `srun`, often with `likwid-pin`) that runs a benchmark executable from `src/` (built in a sibling build directory such as `../../build_gcc_new/src/`) or one of the Python/Julia comparison scripts, sweeping problem sizes or thread counts.
- `<name>.sh-<jobid>.out`: stdout of that job (raw timing/performance statistics printed by `pitts_performance`). `*.log`: same for older runs. `*.out.dat`: extracted tables.
- Python/Julia scripts here call *other* libraries for comparison against PITTS (ttpy, scikit_tt, t3f, tntorch, torchtt, teneva, tensorly, xerus, TensorToolbox, numpy/torch LAPACK, ITensorMPS.jl).
- `.gitignore` ignores `*.out` for new runs; the committed logs are historical (license covered by `.reuse/dep5`).

## Layout

| Path | Content |
|---|---|
| `bench/*.py` | Comparison benchmarks with other libraries: `numpy_{gemm,qr,svd,tt_op_dense}_bench.py`, `numpy_qr_bench_row_major.py`, `torch_{gemm,qr,svd,tt_op_dense}_bench.py`, `opt_einsum_tt_op_dense_bench.py`, `tt_from_dense_{ttpy,t3f,tntorch,torchtt,teneva,tensorly,xerus,TensorToolbox,dgesdd}.py` (TT-SVD in other libraries), `ttpy_amen.py` / `scikit_tt_mals.py` (linear solvers in other libraries), `ttn_contraction_local_gradient_opt.py` (numpy/opt_einsum reference for the TTN local gradient contraction). |
| `bench/*.jl` | `tt_from_dense_ITensorMPS.jl`, `tt_axpby_ITensorMPS.jl`, `tt_op_dense_ITensorMPS.jl`: ITensorMPS.jl comparisons. |
| `bench/*.sh` + `.out` | `ttpy_amen*.sh`, `amd_ttpy_amen*.sh`: job scripts for the ttpy AMEn comparison. |
| `compile/` | Scripts measuring compile time: `build_{gcc,clang}[_ccache|_devbuild].sh` (Ninja, optional ccache, optional `PITTS_DEVELOP_BUILD`). |
| `trilinos/` | Small separate CMake project (`tsqr.cpp`) benchmarking Trilinos/Tpetra TSQR for comparison. |
| `helma/` | TSQR vs MKL QR runs on the "helma" machine. |
| `sc-cluster/` | Main result archive from the DLR SC cluster (Intel and AMD single-socket nodes). Top level: TSQR (`tsqr_single_socket*`), TSMM, GEMM/QR/SVD baselines (numpy, plasma, trilinos), memory bandwidth and peak flops (likwid-bench), TT-SVD in all variants (`tt_from_dense_*`: plain, thickbounds, twosided, autofusing, classical, dgesdd, likwid-instrumented) and against other libraries, truncation/orthogonalization kernels, `tt_axpby_ITensorMPS*`. Helper: `tt_from_dense_single_socket_extract_data_per_function.sh`. |
| `sc-cluster/scaling/`, `scaling2/` | Weak and strong MPI scaling of the thick-bounds TT-SVD (`d<dims>_N<nodes>_n<procs>`, `d32_strong_n<procs>`). |
| `sc-cluster/tt_axpy/` | TT axpby variants (plain, left/right-normalized, with/without TSQR, reorthogonalization) on AMD. |
| `sc-cluster/tt_op_dense/` | TT operator applied to dense vectors: PITTS (with/without direct MKL) vs numpy, torch, ITensor. |
| `sc-cluster/tt_krylov/` | TT-GMRES experiments with preconditioners (rank-1, SSOR, left/right), adaptive vs fixed tolerances; drivers `tt_gmres_precond*.py`, `random_op.py`; subfolder `eps_by_m_tolerance/`. |
| `sc-cluster/tt_mals/` | ALS/MALS/AMEn solver experiments on convection-diffusion problems; driver `tt_mals.py`, extraction scripts `extract_data.py`, `extract_data_amen_als.py`, extracted `*.out.dat`. |
| `sc-cluster/tt_qr/`, `transform_transpose/`, `triangular_solve_mult/`, `tsqr_parallelize/` | Smaller kernel studies (QR of TT cores, TSMM/transpose, triangular solve/mult, TSQR thread-synchronization experiment with `remarks.md`). |
