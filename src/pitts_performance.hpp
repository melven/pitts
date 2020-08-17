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
#include <iomanip>
#include <vector>
#include <numeric>
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
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, double callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
      const internal::ScopeWithArgumentInfo scopeArgs{scope, arguments, callsPerFunction};
      const auto [iter, didInsert] = globalPerformanceStatisticsMap.insert({scopeArgs, {internal::TimingStatistics(), kernel}});
      return internal::ScopedTimer(iter->second.timings);
    }


    //! Measure the runtime of the curent function or scope and gather performance statistics (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::FixedArgumentInfo arguments, kernel_info::KernelInfo kernel, double callsPerFunction = 1, internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(arguments, kernel, callsPerFunction, scope);
    }


    //! print nice statistics using globalPerformanceStatisticsMap
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout)
    {
      // For sorting and nicer formatting, first copy all stuff into an array of small helper structs
      struct Line final
      {
        std::string description;
        double totalTime;
        std::size_t calls;
        double gflops_dp;
        double gflops_sp;
        double gbytes;
      };
      std::vector<Line> lines;
      lines.reserve(globalPerformanceStatisticsMap.size());
      for(const auto& [scopeWithArgs, performanceData]: globalPerformanceStatisticsMap)
      {
        const auto& scope = scopeWithArgs.scope;
        const auto& args = scopeWithArgs.args;
        const auto& timings = performanceData.timings;
        const auto& flops = performanceData.kernel.flops;
        const auto& bytes = performanceData.kernel.bytes;

        Line line;
        if( !std::string_view(scope.type_name()).empty() )
          line.description = scope.type_name() + std::string("::");
        line.description += scope.function_name();
        line.description += "(" + args.to_string() + ")";

        line.totalTime = timings.totalTime;
        line.calls = timings.calls;
        line.gflops_dp = timings.calls*flops.doublePrecision/timings.totalTime*1.e-9;
        line.gflops_sp = timings.calls*flops.singlePrecision/timings.totalTime*1.e-9;
        line.gbytes = timings.calls*(2*bytes.update+bytes.load+bytes.store)/timings.totalTime*1.e-9;

        lines.emplace_back(std::move(line));
      }

      // sort by decreasing time
      std::sort(lines.begin(), lines.end(), [](const Line& l1, const Line& l2){return l1.totalTime > l2.totalTime;});
      // get maximal length of the name string
      const auto maxDescLen = std::accumulate(lines.begin(), lines.end(), 10, [](std::size_t n, const Line& l){return std::max(n, l.description.size());});


      // actual output
      out << "Performance statistics:\n";
      out << std::left;
      out << std::setw(maxDescLen) << "function" << "\t "
          << std::setw(10) << "time [s]" << "\t "
          << std::setw(10) << "#calls" << "\t "
          << std::setw(10) << "GFlop/s DP" << "\t "
          << std::setw(10) << "GFlop/s SP" << "\t "
          << std::setw(10) << "GByte/s" << "\t"
          << std::setw(10) << "Flops/Byte" << "\n";
      for(const auto& line: lines)
      {
        out << std::setw(maxDescLen) << line.description << "\t "
            << std::setw(10) << line.totalTime << "\t "
            << std::setw(10) << line.calls << "\t "
            << std::setw(10) << line.gflops_dp << "\t "
            << std::setw(10) << line.gflops_sp << "\t "
            << std::setw(10) << line.gbytes << "\t"
            << std::setw(10) << (line.gflops_sp+line.gflops_dp) / line.gbytes << "\n";
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
