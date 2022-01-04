# PITTS - Parallel Iterative Tensor-Train Solvers

Small header-only C++20 library for numerical algorithms with low-rank tensor approximations in *tensor train* form (TT format, see [2](README.md#References) and [3](README.md#References)).

Also provides a numpy-compatible python interface based on pybind11.
Currently, algorithms are parallelized for multi-core CPU clusters using OpenMP and MPI.

## Getting Started
You can get a copy of the repository from [https://github.com/melven/pitts]:

    > git clone https://github.com/melven/pitts.git

### Prerequisites
* [CMake](https://cmake.org) >= 3.14 (tested with 3.18.1)
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

    > cd pitts
    > mkdir build
    > cd build
    > cmake .. -DCMAKE_BUILD_TYPE=Release
    > make

### Running the tests
Internally uses [googletest](https://github.com/google/googletest) (with patches for MPI parallelization) for testing C++ code
and python unittest to check the behavior of the python interface.

    > make check

The tests run with different numbers of OpenMP threads and MPI processes.
They call `mpiexec` to launch multiple processes, respectively the SLURM command `srun` when a SLURM queueing system is found.

### Python interface
For simple tests, add the build directory `pitts/build/src` to the `PYTHONPATH` environment variable.
You can also install it in a custom directory using the `CMAKE_INSTALL_PREFIX` setting and type `make install`.

To use PITTS in your python code, simply import `pitts_py`

    > import pitts_py
    > # create a random tensor in TT format with dimensions 2^5 and TT ranks [2,4,4,2]
    > tt = pitts_py.TensorTrain_double(dimensions=[2,2,2,2,2])
    > tt.setTTranks([2,4,4,2])
    > pitts_py.randomize(tt)

## References

[1] Roehrig-Zoellner, M., Thies, J. and Basermann, A.: "Performance of low-rank approximations in tensor train format (TT-SVD) for large dense tensors", submitted to SISC, 2021, https://arxiv.org/abs/2102.00104

[2] Oseledets, I. V.: "Tensor-Train Decomposition", SIAM Journal on Scientific Computing, 2011, https://doi.org/10.1137/090752286

[3] Grasedyck, L., Kressner, D. and Tobler, C.: "A literature survey of low-rank tensor approximation techniques", GAMM-Mitteilungen, 2013, https://doi.org/10.1002/gamm.201310004

