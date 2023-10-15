// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_parallel.hpp
* @brief Helper functionality for parallelization (OpenMP, MPI)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-19
*
**/

// include guard
#ifndef PITTS_PARALLEL_HPP
#define PITTS_PARALLEL_HPP

// includes
#include <mpi.h>
#include <omp.h>

#include <unordered_map>
#include <numeric>
#include <vector>
#include <string>
#include <string_view>
#include <functional>
#include <tuple>
#include <type_traits>
#include <complex>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    // small helper type that defers static_assert evaluation to template instantiation
    // (copied from https://www.fluentcpp.com/2019/08/23/how-to-make-sfinae-pretty-and-robust/)
    template<typename>
    inline constexpr bool dependent_false_v{ false };

    //! helper functionality for distribution of data and parallelization
    namespace parallel
    {
      //! OpenMP helper: get index of the current thread and the total number of threads
      //!
      //! @warning This must be called from inside an OpenMP parallel region.
      //!          In an OpenMP-serial scope, it returns iThread=0, nThreads=1
      //!
      //! @returns  pair(iThread,nThreads)
      //!
      inline auto ompThreadInfo()
      {
        const auto iThread = omp_get_thread_num();
        const auto nThreads = omp_get_num_threads();
        return std::make_pair(iThread, nThreads);
      }

      //! MPI helper: get index of the curreant process and the total number of processes
      //!
      //! @param comm   MPI communicator
      //!
      //! @returns  pair(iProc,nProcs)
      //!
      inline auto mpiProcInfo(MPI_Comm comm = MPI_COMM_WORLD)
      {
        int iProc = 0, nProcs = 1;

        if( MPI_Comm_rank(comm, &iProc) != MPI_SUCCESS )
          throw std::runtime_error("failure returned from MPI_Comm_rank");

        if( MPI_Comm_size(comm, &nProcs) != MPI_SUCCESS )
          throw std::runtime_error("failure returned from MPI_Comm_rank");

        return std::make_pair(iProc, nProcs);
      }

      //! Calculate offsets to distribute a given number of elements equally on the given number of processing units
      //!
      //! This can be used for both MPI and OpenMP (for cases where the loop structure inhibits just using `#pragma omp for schedule(static)`)
      //!
      //! @warning This function returns an index range (firstElem,lastElem) in the form (offset,offset+nLocal-1) where lastElem is actually the last index that should be handled.
      //!          For fewer elements than processing units, some processing units don't have any work to do, so in that case lastElem=firstElem-1.
      //!
      //! @param nElems             global number of elements to distribute
      //! @param procIndexAndTotal  pair(i,n) where 0<=i<n is the index of the current processing unit and n is the number of processing units
      //! @returns                  pair(firstElem,lastElem); index range of elements that should be handled by this processing unit
      //!
      constexpr auto distribute(long long nElems, const std::pair<int,int>& procIndexAndTotal) noexcept
      {
        const auto& [iProc,nProcs] = procIndexAndTotal;
        long long firstElem = 0;
        long long lastElem = nElems - 1;
        if( nProcs > 1 )
        {
          long long nElemsPerThread = nElems / nProcs;
          long long nElemsModThreads = nElems % nProcs;
          if( iProc < nElemsModThreads )
          {
            firstElem = iProc * (nElemsPerThread+1);
            lastElem = firstElem + nElemsPerThread;
          }
          else
          {
            firstElem = iProc * nElemsPerThread + nElemsModThreads;
            lastElem = firstElem + nElemsPerThread-1;
          }
        }
        return std::make_pair(firstElem, lastElem);
      }

      //! Return corresponding MPI data type
      //!
      //! @warning Only implemented for a few commonly used types
      //!
      //! @tparam T   C++ data type
      //! @returns    MPI data type
      //!
      template<typename T>
      constexpr MPI_Datatype mpiType() noexcept
      {
        if constexpr( std::is_same<T, double>:: value )
          return MPI_DOUBLE;
        else if constexpr( std::is_same<T, float>::value )
          return MPI_FLOAT;
        else if constexpr( std::is_same<T, std::complex<double>>::value )
          return MPI_C_DOUBLE_COMPLEX;
        else if constexpr( std::is_same<T, std::complex<float>>::value )
          return MPI_C_FLOAT_COMPLEX;
        else
          static_assert( dependent_false_v<T>, "Desired type is not implemented!");
      }


      //! gather "raw" data of arbitrary length from multiple MPI processes
      //!
      //! @param localData    part of the data of the current process
      //! @param root         (optional) MPI root process, must be identical on all MPI processes
      //! @param comm         (optional) MPI communicator, must be identical on all MPI processes
      //! @returns            pair(data,offsets) with the gathered data from process i at data[offset[i]...offset[i+1]-1] on the root process (otherwise empty)
      //!
      std::pair<std::string,std::vector<int>> mpiGather(const std::string_view& localData, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);


      //! combine std::unordered_map from multiple MPI processes
      //!
      //! @tparam Key               std::unordered_map key type, must be serializable using cereal
      //! @tparam Value             std::unordered_map value type, must be serializable using cereal
      //! @tparam BinaryOperation   function signature to combine two values, defaults to + operator
      //!
      //! @param localMap           std::unordered_map on each process
      //! @param op                 (optional) operator to combine values of keys that occur on multiple processes, only relevant on process 0
      //! @param root               (optional) MPI root process, must be identical on all MPI processes
      //! @param comm               (optional) MPI communicator, must be identical on all MPI processes
      //! @returns                  combined map of all process on the root process, otherwise an empty
      //!
      template<typename Key, typename Value, class BinaryOperation = decltype(std::plus())>
      std::unordered_map<Key,Value> mpiCombineMaps(const std::unordered_map<Key,Value>& localMap, BinaryOperation op = std::plus(), int root = 0, MPI_Comm comm = MPI_COMM_WORLD);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_parallel_impl.hpp"
#endif

#endif // PITTS_PARALLEL_HPP
