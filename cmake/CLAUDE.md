<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# cmake/ — CMake modules

Added to `CMAKE_MODULE_PATH` by the root `CMakeLists.txt`.

| File | Content |
|---|---|
| `GTest.cmake` | `ExternalProject` that unpacks and builds `third-party/gtest-1.8.0-mpi.tar.gz`; sets `GTEST_SOURCES`/`GTEST_INCLUDE_PATH` used by `test/CMakeLists.txt`. |
| `CodeCoverage.cmake` | Option `CODE_COVERAGE` (GCC + Debug only): adds `--coverage` flags and a gcovr-based coverage target (`setup_target_for_coverage`). |
| `prepare_gcovr.sh` | Helper for coverage: renames mangled `#...#.gcda` files so gcovr finds them. |
| `FindLIKWID.cmake` | Finds likwid (`LIKWID_INCLUDE_DIRS`, `LIKWID_LIBRARIES`); optional. |
| `FindITensor.cmake` | Finds ITensor (`ITENSOR_INCLUDE_DIRS`, `ITENSOR_LIBRARIES`); optional. |
| `GetGitRevisionDescription.cmake`, `.cmake.in` | Third-party module (rpavlik) providing `git_describe_working_tree`, used for `PITTS_GIT_VERSION`. |
