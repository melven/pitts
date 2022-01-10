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
#include "pitts_parallel.hpp"
#include "pitts_kernel_info.hpp"
#include "pitts_timer.hpp"
#include <cmath>
#include <iomanip>
#include <vector>
#include <numeric>
#include <string>


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


    //! helper type for performance results with a name
    struct NamedPerformance final
    {
      //! description (function name, parameters, etc)
      std::string name;

      //! performance results
      PerformanceStatistics performance;
    };

    //! gather performance data in a vector, so they can be sorted and printed more easily
    inline auto gatherPerformance(const PerformanceStatisticsMap& map, bool mpiGlobal = true)
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
      {
        // operator that only adds up timings
        const auto combineOp = [](const PerformanceStatistics& a, const PerformanceStatistics& b)
        {
          // check that kernel data matches!!
          if( a.kernel != b.kernel )
            throw std::invalid_argument("Trying to combine timings of functions with different performance characteristics (Flops, Bytes)!");
          return PerformanceStatistics{a.timings+b.timings, a.kernel, a.nProcs+b.nProcs};
        };
        namedMap = parallel::mpiCombineMaps(namedMap, combineOp);
      }

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
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout, bool mpiGlobal = true)
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


#endif // PITTS_PERFORMANCE_HPP
