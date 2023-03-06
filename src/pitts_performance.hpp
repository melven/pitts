/*! @file pitts_performance.hpp
* @brief Helper functionality for measuring the performance of C++ functions / scopes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-05-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_PERFORMANCE_HPP
#define PITTS_PERFORMANCE_HPP

// includes
#include "pitts_kernel_info.hpp"
#include "pitts_timer.hpp"
#include <stdexcept>
#include <unordered_map>
#include <iostream>

#ifdef PITTS_DEVELOP_BUILD
#include "pitts_missing_cereal.hpp"
#else
#include <cereal/cereal.hpp>
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper for accumulating timings and storing required flops/bytes per call
    struct PerformanceStatistics final
    {
      //! measured run-time data
      TimingStatistics timings;

      //! theoretical information on required operations and data transfers
      kernel_info::KernelInfo kernel;

      //! number of processing units involved (e.g. MPI processes)
      int nProcs = 1;

      //! allow reading/writing with cereal
      template<class Archive>
      void serialize(Archive & ar)
      {
        ar( CEREAL_NVP(timings),
            CEREAL_NVP(kernel),
            CEREAL_NVP(nProcs) );
      }

      // operator that only adds up timings
      PerformanceStatistics operator+(const PerformanceStatistics& other)
      {
        // check that kernel data matches!!
        if( kernel != other.kernel )
          throw std::invalid_argument("Trying to combine timings of functions with different performance characteristics (Flops, Bytes)!");
        return PerformanceStatistics{timings+other.timings, kernel, nProcs+other.nProcs};
      };
    };


    //! helper type for storing timings with performance data per function / scope and per argumen
    using PerformanceStatisticsMap = std::unordered_map<internal::ScopeWithArgumentInfo, internal::PerformanceStatistics, internal::ScopeWithArgumentInfo::Hash>;

    //! combine timings with different arguments for each function
    TimingStatisticsMap combineTimingsPerFunction(const PerformanceStatisticsMap& performanceStats);

    //! gather performance data in a vector, so they can be sorted and printed more easily
    auto gatherPerformance(const PerformanceStatisticsMap& map, bool mpiGlobal = true);
  }


  //! namespace for runtime measurement data of performance statistics (timings + #flops / data transfered)
  namespace performance
  {
    //! container for storing timings for different arguments with createScopedTimer
    inline internal::PerformanceStatisticsMap globalPerformanceStatisticsMap;


    //! Measure the runtime of the curent function or scope and gather performance statistics
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, double callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
      const internal::ScopeWithArgumentInfo scopeArgs{scope, arguments, callsPerFunction};
      const auto [iter, didInsert] = globalPerformanceStatisticsMap.insert({scopeArgs, {internal::TimingStatistics(), kernel}});
#ifndef PITTS_USE_LIKWID_MARKER_API
      return internal::ScopedTimer(iter->second.timings);
#else
      return std::tuple<internal::ScopedTimer, internal::ScopedLikwidRegion>{
          iter->second.timings,
          scope.function_name()};
#endif
    }


    //! Measure the runtime of the curent function or scope and gather performance statistics (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, double callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(arguments, kernel, callsPerFunction, scope);
    }


    //! print nice statistics using globalPerformanceStatisticsMap
    void printStatistics(bool clear = true, std::ostream& out = std::cout, bool mpiGlobal = true);
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_performance_impl.hpp"
#endif

#endif // PITTS_PERFORMANCE_HPP
