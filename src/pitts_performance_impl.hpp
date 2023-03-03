/*! @file pitts_performance_impl.hpp
* @brief Helper functionality for measuring the performance of C++ functions / scopes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-05-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_PERFORMANCE_IMPL_HPP
#define PITTS_PERFORMANCE_IMPL_HPP

// includes
#include <cmath>
#include <iomanip>
#include <vector>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include "pitts_performance.hpp"

// workaround for speeding up compile times during development
#ifndef PITTS_DEVELOP_BUILD
#define INLINE inline
#else
#define INLINE
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    INLINE TimingStatisticsMap combineTimingsPerFunction(const PerformanceStatisticsMap& performanceStats)
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


    //! helper type for performance results with a name
    struct NamedPerformance final
    {
      //! description (function name, parameters, etc)
      std::string name;

      //! performance results
      PerformanceStatistics performance;
    };

    //! gather performance data in a vector, so they can be sorted and printed more easily
    INLINE auto gatherPerformance(const PerformanceStatisticsMap& map, bool mpiGlobal)
    {
      // copy to serializable map (ScopInfo is read-only)
      std::unordered_map<std::string,PerformanceStatistics> namedMap;
      for(const auto& [scopeWithArgs, performance]: map)
      {
        const auto& scope = scopeWithArgs.scope;
        const auto& args = scopeWithArgs.args;

        std::string fullName;
        if( !std::string_view(scope.type_name()).empty() )
          fullName.append("<").append(scope.type_name()).append("> :: ");
        fullName.append(scope.function_name());
        fullName.append("(" + args.to_string() + ")");

        namedMap.insert({std::move(fullName),performance});
      }

      if( mpiGlobal )
        namedMap = parallel::mpiCombineMaps(namedMap);

      std::vector<NamedPerformance> result;
      result.reserve(namedMap.size());

      for(const auto& [name, performance]: namedMap)
        result.emplace_back(NamedPerformance{name, performance});

      return result;
    }
  }


  //! namespace for runtime measurement data of performance statistics (timings + #flops / data transfered)
  namespace performance
  {
    // implement statistics printer
    INLINE void printStatistics(bool clear, std::ostream& out, bool mpiGlobal)
    {
      using internal::NamedPerformance;
      std::vector<NamedPerformance> lines = internal::gatherPerformance(globalPerformanceStatisticsMap, mpiGlobal);

      // sort by decreasing time
      std::sort(lines.begin(), lines.end(), [](const NamedPerformance& l1, const NamedPerformance& l2){return l1.performance.timings.totalTime > l2.performance.timings.totalTime;});
      // get maximal length of the name string
      const auto maxDescLen = std::accumulate(lines.begin(), lines.end(), 10, [](std::size_t n, const NamedPerformance& l){return std::max(n, l.name.size());});

      // prevent output on non-root with mpiGlobal == true (but omit mpiProcInfo call when mpiGlobal == false)
      bool doOutput = true;

      if( mpiGlobal )
      {
        const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
        doOutput = iProc == 0;
      }

      if( doOutput )
      {
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
          const auto& timings = line.performance.timings;
          const auto& nProcs = line.performance.nProcs;
          const auto& flops = line.performance.kernel.flops;
          const auto& bytes = line.performance.kernel.bytes;
          out << std::setw(maxDescLen) << line.name << "\t "
              << std::setw(10) << timings.totalTime/nProcs << "\t "
              << std::setw(10) << std::lround(timings.calls*1.0/nProcs) << "\t "
              << std::setw(10) << timings.calls*flops.doublePrecision/(timings.totalTime/nProcs)*1.e-9 << "\t "
              << std::setw(10) << timings.calls*flops.singlePrecision/(timings.totalTime/nProcs)*1.e-9 << "\t "
              << std::setw(10) << timings.calls*(2*bytes.update+bytes.load+bytes.store)/(timings.totalTime/nProcs)*1.e-9 << "\t "
              << std::setw(10) << (flops.doublePrecision+flops.singlePrecision) / (2*bytes.update+bytes.load+bytes.store) << "\n";
        }
      }


      // also print timing statistics
      if( clear )
      {
        timing::globalTimingStatisticsMap += combineTimingsPerFunction(globalPerformanceStatisticsMap);
        timing::printStatistics(true, out);

        globalPerformanceStatisticsMap.clear();
      }
      else
      {
        // no clearing, need to restore globalTimingStatisticsMap after the call
        auto timingStatisticsMap = timing::globalTimingStatisticsMap;

        timing::globalTimingStatisticsMap += combineTimingsPerFunction(globalPerformanceStatisticsMap);

        timing::printStatistics(false, out);
        std::swap(timingStatisticsMap, timing::globalTimingStatisticsMap);
      }
    }
  }
}


#endif // PITTS_PERFORMANCE_IMPL_HPP
