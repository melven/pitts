/*! @file pitts_timer_impl.hpp
* @brief Helper functionality for measuring the runtime of C++ functions / scopes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-04-14
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TIMER_IMPL_HPP
#define PITTS_TIMER_IMPL_HPP

// includes
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <string>
#include <cereal/cereal.hpp>
#include "pitts_timer.hpp"
#include "pitts_parallel.hpp"

#ifdef PITTS_USE_LIKWID_MARKER_API
#include <likwid.h>
#endif

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
    //! helper type for a timing result with a name
    struct NamedTiming final
    {
      //! description (function name, parameters, etc)
      std::string name;

      //! timing results
      TimingStatistics timings;
    };

    //! gather timings in a vector, so they can be sorted and printed more easily
    INLINE auto gatherTimings(const TimingStatisticsMap& map, bool mpiGlobal)
    {
      // copy to serializable map (ScopeInfo is read-only)
      std::unordered_map<std::string,TimingStatistics> namedMap;
      for(const auto& [scope, timings]: map)
      {
        std::string fullName;
        if( !std::string_view(scope.type_name()).empty() )
          fullName.append("<").append(scope.type_name()).append(std::string("> :: "));
        fullName.append(scope.function_name());

        namedMap.insert({std::move(fullName),timings});
      }

      if( mpiGlobal )
        namedMap = parallel::mpiCombineMaps(namedMap);

      std::vector<NamedTiming> result;
      result.reserve(namedMap.size());
      for(const auto& [name, timings]: namedMap)
        result.emplace_back(NamedTiming{name, timings});

      return result;
    }
  }


  //! namespace for runtime measurement data and helper functions
  namespace timing
  {
    //! print nice statistics using globalTimingStatisticsMap
    INLINE void printStatistics(bool clear, std::ostream& out, bool mpiGlobal)
    {
      using internal::NamedTiming;
      std::vector<NamedTiming> lines = gatherTimings(globalTimingStatisticsMap, mpiGlobal);

      // sort by decreasing time
      std::sort(lines.begin(), lines.end(), [](const NamedTiming& l1, const NamedTiming& l2){return l1.timings.totalTime > l2.timings.totalTime;});
      // get maximal length of the name string
      const auto maxNameLen = std::accumulate(lines.begin(), lines.end(), 10, [](std::size_t n, const NamedTiming& l){return std::max(n, l.name.size());});

      // prevent output on non-root with mpiGlobal == true (but omit mpiProcInfo call when mpiGlobal == false)
      bool doOutput = true;

      if( mpiGlobal )
      {
        // divide data by number of processes (average timings/calls)
        const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
        for(auto& line: lines)
        {
          line.timings.totalTime /= nProcs;
          line.timings.calls /= nProcs;
        }

        doOutput = iProc == 0;
      }

      if( doOutput )
      {
        // actual output
        out << "Timing statistics:\n";
        out << std::left;
        out << std::setw(maxNameLen) << "function" << "\t "
            << std::setw(10) << "time [s]" << "\t "
            << std::setw(10) << "#calls" << "\n";
        for(const auto& line: lines)
        {
          out << std::setw(maxNameLen) << line.name << "\t "
              << std::setw(10) << line.timings.totalTime << "\t "
              << std::setw(10) << line.timings.calls << "\n";
        }
      }

      if( clear )
        globalTimingStatisticsMap.clear();
    }
  }
}


#endif // PITTS_TIMER_IMPL_HPP
