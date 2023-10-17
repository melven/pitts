<!--
SPDX-FileCopyrightText: 2023 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# PITTS

[![REUSE status](https://api.reuse.software/badge/github.com/melven/pitts)](https://api.reuse.software/info/github.com/melven/pitts)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

PITTS--Parallel Iterative Tensor-Train Solvers--is a small header-only C++20 library for numerical algorithms with low-rank tensor approximations in *tensor train* form (TT format, see [2](#references) and [3](#references)).
It also provides a numpy-compatible python interface based on pybind11. Algorithms are parallelized for multi-core CPU clusters using OpenMP and MPI.

Currently provides a fast TT-SVD implementation (algorithm to compress a dense tensor in the TT format), and methods for solving linear systems (symmetric and non-symmetric) in tensor-train format (TT-GMRES [4](#references), TT-MALS [5](#references), TT-AMEn [6](#references)).

## Table of Contents

- [Install](#install)
	- [Dependencies](#dependencies)
	- [Compiling](#compiling)
	- [Running the tests](#running-the-tests)
- [Usage](#usage)
	- [C++ interface](#c-interface)
	- [Python interface](#python-interface)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Install
You can get a copy of the repository from [https://github.com/melven/pitts]:

```sh
git clone https://github.com/melven/pitts.git
```

### Dependencies
* [CMake](https://cmake.org) >= 3.18 (tested with 3.18.1)
* [GCC](https://gcc.gnu.org) >= 11.1 (or a C++20 compliant compiler)
  * [OpenMP](https://www.openmp.org) (usually included in the compiler)
* [MPI](https://www.mpi-forum.org) (tested with [OpenMPI](https://open-mpi.org) 4.0)
* [LAPACK](http://www.netlib.org/lapack) (tested with [Intel MKL](https://software.intel.com/en-us/intel-mkl) 2020)
* [Eigen](https://eigen.tuxfamily.org) >= 3.3.9 (3.3.8 has a C++20 bug!)
* [cereal](https://uscilab.github.io/cereal) (tested with 1.3.0)
* [pybind11](https://github.com/pybind/pybind11) (tested with 2.5.0)
* [Python](https://www.python.org) >= 3.6 (tested with 3.8.3)

### Compiling
Simply configure with CMake and compile, on Linux system usually done by:

```sh
cd pitts
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Running the tests
Internally uses [googletest](https://github.com/google/googletest) [with patches for MPI parallelization](https://github.com/DLR-SC/googletest_mpi) for testing C++ code
and python unittest to check the behavior of the python interface.

```sh
make check
```

The tests run with different numbers of OpenMP threads and MPI processes.
They call `mpiexec` to launch multiple processes, respectively the SLURM command `srun` when a SLURM queueing system is found.

## Usage
Currently, pitts can be used from C++ and python.

### C++ interface
pitts is intended as header-only library and its data types and algorithms can be directly included:

```c++
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"

// create a random tensor in TT format with dimensions 2^5 and TT ranks [2,4,4,2]
void someFunction()
{
  PITTS::TensorTrain<double> tt(2,5);
  tt.setTTranks({2,4,4,2});
  // most algorithms in PITTS are defined as free functions with overloads for different data types,
  // this calls PITTS::randomize(PITTS::TensorTrain<double>&)
  randomize(tt);
}
```

As pitts heavily uses templates and C++20 features, using pitts from C++ code requires a C++20 compliant compiler (and enabling C++20, of course).

### Python interface
For simple tests, add the build directory `pitts/build/src` to the `PYTHONPATH` environment variable.
You can also install it in a custom directory using the `CMAKE_INSTALL_PREFIX` setting and type `make install`.

To use PITTS in your python code, simply import `pitts_py`:

```python
import pitts_py
# create a random tensor in TT format with dimensions 2^5 and TT ranks [2,4,4,2]
tt = pitts_py.TensorTrain_double(dimensions=[2,2,2,2,2])
tt.setTTranks([2,4,4,2])
pitts_py.randomize(tt)
```

## References

[1] Roehrig-Zoellner, M., Thies, J. and Basermann, A.: "Performance of the Low-Rank TT-SVD for Large Dense Tensors on Modern MultiCore CPUs", SIAM Journal on Scientific Computing, 2022, https://doi.org/10.1137/21M1395545

[2] Oseledets, I. V.: "Tensor-Train Decomposition", SIAM Journal on Scientific Computing, 2011, https://doi.org/10.1137/090752286

[3] Grasedyck, L., Kressner, D. and Tobler, C.: "A literature survey of low-rank tensor approximation techniques", GAMM-Mitteilungen, 2013, https://doi.org/10.1002/gamm.201310004

[4] Dolgov, S. V.: "TT-GMRES: solution to a linear system in the structured tensor format", Russian Journal of Numerical Analysis and Mathematical Modelling, 2013, https://doi.org/10.1515/rnam-2013-0009

[5] Holtz, S., Rohwedder, T. and Schneider, R.: "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing, 2012, https://doi.org/10.1137/100818893

[6] Dolgov, S. V. and Savostyanov, D. V.: "Alternating Minimal Energy Methods for Linear Systems in Higher Dimensions" SIAM Journal on Scientific Computing, 2014, http://doi.org/10.1137/140953289

## Contributing

Please feel free to send any question or suggestion to Melven.Roehrig-Zoellner@DLR.de.

## License

[BSD-3 Clause](LICENSE)
