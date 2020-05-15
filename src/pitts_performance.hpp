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
#include <cmath>
#include "pitts_kernel_info.hpp"
#include "pitts_timer.hpp"


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
    };


    //! helper type for storing timings with performance data per function / scope and per argumen
    using PerformanceStatisticsMap = std::unordered_map<internal::ScopeWithArgumentInfo, internal::PerformanceStatistics, internal::ScopeWithArgumentInfo::Hash>;

    //! combine timings with different arguments for each function
    inline TimingStatisticsMap combineTimingsPerFunction(const PerformanceStatisticsMap& performanceStats)
    {
      TimingStatisticsMap timingStats;

      for(const auto& [scopeWithArgs, performanceData]: performanceStats)
      {
        timingStats[scopeWithArgs.scope] += performanceData.timings;
      }

      // call counts are more tricky as we allow multiple "sub-scopes" in a single function
      // for a more detailed performance analysis with low overhead (during measurement)
      // so replace call counts by corrected call counts
      std::unordered_map<internal::ScopeInfo, double, internal::ScopeInfo::Hash> functionCallCounts;
      // sum up "weighted" function call counts (divided by number of "sub-scopes" per function)
      for(const auto& [scopeWithArgs, performanceData]: performanceStats)
      {
        functionCallCounts[scopeWithArgs.scope] += 1./scopeWithArgs.callsPerFunction * performanceData.timings.calls;
      }
      // replace call counts with corrected data
      for(const auto& [scope, calls]: functionCallCounts)
      {
        timingStats[scope].calls = std::lround(calls);
      }

      return timingStats;
    }
  }


  //! namespace for runtime measurement data of performance statistics (timings + #flops / data transfered)
  namespace performance
  {
    //! container for storing timings for different arguments with createScopedTimer
    inline internal::PerformanceStatisticsMap globalPerformanceStatisticsMap;


    //! Measure the runtime of the curent function or scope and gather performance statistics
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, int callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
      const internal::ScopeWithArgumentInfo scopeArgs{scope, arguments, callsPerFunction};
      const auto [iter, didInsert] = globalPerformanceStatisticsMap.insert({scopeArgs, {internal::TimingStatistics(), kernel}});
      return internal::ScopedTimer(iter->second.timings);
    }


    //! Measure the runtime of the curent function or scope and gather performance statistics (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, int callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(arguments, kernel, callsPerFunction, scope);
    }


    //! print nice statistics using globalPerformanceStatisticsMap
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout)
    {
      out << "Performance statistics:\n";

      // header
      out << "function:\t time [s] (#calls)\t GFlop/s DP\t GFlop/s SP\t GByte/s\n";
      for(const auto& [scopeWithArgs, performanceData]: globalPerformanceStatisticsMap)
      {
        const auto& scope = scopeWithArgs.scope;
        const auto& args = scopeWithArgs.args;
        const auto& timings = performanceData.timings;
        const auto& flops = performanceData.kernel.flops;
        const auto& bytes = performanceData.kernel.bytes;

        // print (optional) type
        if( !std::string_view(scope.type_name()).empty() )
          out << scope.type_name() << "::";
        // print function
        out << scope.function_name();

        // print arguments
        out << "(" << args.to_string() << ")";

        // timings
        out << " : " << timings.totalTime << " (" << timings.calls << ")";

        // floating point operations (double-precision GFlop/s)
        out << "\t " << timings.calls*flops.doublePrecision/timings.totalTime*1.e-9;

        // floating point operations (single-precision GFlop/s)
        out << "\t " << timings.calls*flops.singlePrecision/timings.totalTime*1.e-9;

        // data transfers (GByte/s)
        out << "\t " << timings.calls*(2*bytes.update+bytes.load+bytes.store)/timings.totalTime*1.e-9;

        out << "\n";
      }


      // also print timing statistics
      auto timingStatisticsMap = timing::globalTimingStatisticsMap;

      timing::globalTimingStatisticsMap = timingStatisticsMap + combineTimingsPerFunction(globalPerformanceStatisticsMap);

      timing::printStatistics(clear, out);
      std::swap(timingStatisticsMap, timing::globalTimingStatisticsMap);

      if( clear )
        globalPerformanceStatisticsMap.clear();
    }
  }
}


#endif // PITTS_PERFORMANCE_HPP
