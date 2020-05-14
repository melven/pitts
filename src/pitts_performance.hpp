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


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  /*
    //! print nice statistics using globalTimingStatisticsMap
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout)
    {
      out << "Timing statistics:\n";
      for(const auto& [scope, timings]: globalTimingStatisticsMap)
      {
        if( !std::string_view(scope.type_name()).empty() )
          out << scope.type_name() << " :: " << scope.function_name() << " : " << timings.totalTime << " (" << timings.calls << ")\n";
        else
          out << scope.function_name() << " : " << timings.totalTime << " (" << timings.calls << ")\n";
      }

      if( clear )
        globalTimingStatisticsMap.clear();
    }
  */

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
  }


  //! namespace for runtime measurement data of performance statistics (timings + #flops / data transfered)
  namespace performance
  {
    //! container for storing timings for different arguments with createScopedTimer
    inline internal::PerformanceStatisticsMap globalPerformanceStatisticsMap;


    //! Measure the runtime of the curent function or scope and gather performance statistics
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
      const internal::ScopeWithArgumentInfo scopeArgs{scope, arguments};
      const auto [iter, didInsert] = globalPerformanceStatisticsMap.insert({scopeArgs, {internal::TimingStatistics(), kernel}});
      return internal::ScopedTimer(iter->second.timings);
    }


    //! Measure the runtime of the curent function or scope and gather performance statistics (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(arguments, kernel, scope);
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

      if( clear )
        globalPerformanceStatisticsMap.clear();
    }
  }
}


#endif // PITTS_PERFORMANCE_HPP
