<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# Notes for AI coding agents (Claude Code etc.)

PITTS is a header-only C++20 library (OpenMP + MPI) with a pybind11 Python binding.
See `README.md` for the project description and `.gitlab-ci.yml` for the reference CI setup.

## Repository layout

Each main directory has its own `CLAUDE.md` with per-file summaries; read it before working there.

| Path | Content |
|---|---|
| `src/` | Library headers (`pitts_*.hpp` + `*_impl.hpp`), explicit-instantiation `.cpp` files for `PITTS_DEVELOP_BUILD`, Python/Julia bindings, benchmark executables. See `src/CLAUDE.md`. |
| `test/` | googletest (MPI-patched) C++ unit tests, Python `unittest` tests for `pitts_py`, Julia tests, `itensor/` subfolder. See `test/CLAUDE.md`. |
| `examples/` | Python example scripts (TT solvers/preconditioners prototyped on top of `pitts_py`). See `examples/CLAUDE.md`. |
| `bench/` | SLURM scripts, comparison scripts for other TT libraries and ~2.4 GB of archived logs. Only a high-level `bench/CLAUDE.md`; do not scan the logs. |
| `cmake/` | CMake modules (googletest unpacking, coverage, Find* modules, git version). See `cmake/CLAUDE.md`. |
| `third-party/` | `gtest-1.8.0-mpi.tar.gz` (googletest with MPI patch) and its license file. |
| `LICENSES/` | License texts (BSD-3-Clause for PITTS, BSL-1.0) for REUSE compliance; `.reuse/dep5` covers files without headers (bench logs/scripts, .gitignore). |
| `CMakeLists.txt` | Root build: compiler flags (`-march=native`, GCC inliner params), options `PITTS_DEVELOP_BUILD`, `PITTS_OPTIMIZE_FOR_HOST`, `PITTS_EIGEN_USE_LAPACKE`, `PITTS_DIRECT_MKL_GEMM`, algorithm-variant switches (`PITTS_TENSORTRAIN_*`), required/optional dependency lookup. |
| `.gitlab-ci.yml` | CI jobs: release/debug/coverage/clang builds with ccache and MKL modules, REUSE lint, GitLab pages publishing the coverage report. |
| `CITATION.cff`, `README.md`, `LICENSE` | Citation metadata (Zenodo DOI), project description, license. |
| `DESIGN.md` | Design goals and conventions (API shape, kernels, TT specifics, testing). Read before adding or changing library code. Coding style is in `src/CLAUDE.md`. |
| `.claude/` | `settings.json` registers `hooks/session-start.sh`, which performs the whole environment setup below for Claude Code on the web. |

Every new source or documentation file needs an SPDX header (REUSE compliance is checked in CI; check locally with `reuse lint`, which the hook installs). For files that cannot carry a header (e.g. JSON) add a `<file>.license` file next to it.

## Environment setup (Ubuntu 24.04, apt only)

In Claude Code on the web the SessionStart hook (`.claude/hooks/session-start.sh`) already did
all of this: packages installed, test environment variables exported, `build/` configured and
built. Check with `ls build/test/pitts_tests`; if it is missing, run the hook manually:
`CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`.

Install everything through the Ubuntu package manager; do not mix in pip packages.

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ccache gdb lld reuse \
  libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev \
  libeigen3-dev libcereal-dev pybind11-dev python3.12-dev python3-numpy
```

Notes:
* If `apt-get update` fails on a third-party PPA (403/unsigned), move its file out of
  `/etc/apt/sources.list.d/` instead of ignoring the error.
* Ubuntu's `python3-numpy` belongs to `/usr/bin/python3.12`. If `python3` on `PATH` is a
  different interpreter (e.g. `/usr/local/bin/python3`), pass `-DPYTHON_EXECUTABLE=/usr/bin/python3.12`
  to CMake so the Python tests use the interpreter that has numpy.
* Optional dependencies not available via apt and therefore skipped: ITensor, JlCxx (Julia), likwid.

## Configure and build

ccache is used exactly as in `.gitlab-ci.yml` (`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`).
The default for agents is the **development build** (`PITTS_DEVELOP_BUILD=On`): templates are
instantiated once in `src/pitts_*.cpp` (object library `pitts_impl`, linked with `lld`), so a
change in one `_impl.hpp` recompiles only that file instead of every test and benchmark.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
  -DPITTS_DEVELOP_BUILD=On
cmake --build build -j$(nproc) --target all pitts_tests   # pitts_tests is EXCLUDE_FROM_ALL
ccache -s                                                # optional: cache statistics
```

A full build takes several minutes on 4 cores; run it in the background and wait for it.
`build*/` directories are git-ignored.

Development-build pitfalls:
* Every template function used outside its own `_impl.hpp` needs an explicit instantiation
  in the matching `src/pitts_<x>.cpp` for all four scalar types; otherwise you get undefined
  references at link time.
* Include sets differ between the two modes (`.hpp` does not pull in `_impl.hpp`). When you
  add or change includes, also compile the affected file in a non-development configuration
  (e.g. `cmake -S . -B build_nodev ... -DPITTS_DEVELOP_BUILD=Off`, then build only the
  affected target or the test executable). CI builds both variants.

## Run the tests

Tests are registered in `build/test` (not in `build/`). The C++ tests run via `mpiexec`
with 1 and 3 MPI processes and 4 OpenMP threads each (`PITTS_TEST_NTHREADS`, keep it at 4:
several tests require at least 4 threads).

```sh
cd build/test
# the session-start hook already exports these; set them yourself otherwise:
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1   # only needed when running as root
export OMP_WAIT_POLICY=passive OPENBLAS_NUM_THREADS=1
ctest --timeout 1200 --output-on-failure
```

Equivalent: `make check` in `build/` (runs ctest with `-V`). These variables are deliberately
not baked into `test/CMakeLists.txt`: the test setup there is tuned for HPC clusters with MKL.

Pitfalls:
* Without `OMP_WAIT_POLICY=passive` (and `OPENBLAS_NUM_THREADS=1`) the 3-process run
  oversubscribes a small machine (3 procs x 4 OpenMP threads + OpenBLAS threads on 4 cores)
  and appears to hang for many minutes. This is not an MPI deadlock; check with
  `gdb -p <pid> -batch -ex "thread 1" -ex bt` before assuming one.
* Expected timings on 4 cores: `unittests_np1` ~10 s, `unittests_np3` ~13 s, Python tests < 6 s each.
* Failures of the form `Expected: (4) <= (omp_get_max_threads())` mean the thread count
  was reduced, not a real bug.

## Running a single C++ test

ctest sets `OMP_NUM_THREADS=4` and `OMPI_MCA_hwloc_base_binding_policy=none` for each test;
set them yourself when calling `mpiexec` directly, otherwise OpenMPI pins the process to one
core and OpenMP sees only 1 thread.

```sh
cd build/test
export OMP_NUM_THREADS=4 OMPI_MCA_hwloc_base_binding_policy=none
mpiexec -n 1 ./pitts_tests --gtest_filter='PITTS_TensorTrain_solve_gmres*'
```
