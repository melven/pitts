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


  }
}


#endif // PITTS_PERFORMANCE_HPP
