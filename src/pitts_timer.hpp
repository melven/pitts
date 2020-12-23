/*! @file pitts_timer.hpp
* @brief Helper functionality for measuring the runtime of C++ functions / scopes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-04-14
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TIMER_HPP
#define PITTS_TIMER_HPP

// includes
#include "pitts_parallel.hpp"
#include "pitts_scope_info.hpp"
#include <chrono>
#include <limits>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <string>

#ifdef PITTS_USE_LIKWID_MARKER_API
#include <likwid.h>
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Helper type for accumulating the timings of one scope / function
    struct TimingStatistics final
    {
      //! sum of all measured timings [s]
      double totalTime = 0;

      //! minimum of all measured timings [s]
      double minTime = std::numeric_limits<double>::max();

      //! maximum of all measured timings [s]
      double maxTime = std::numeric_limits<double>::lowest();

      //! number of measurements
      std::size_t calls = 0;

      //! add another timing measurement to the timing statistics
      constexpr TimingStatistics& operator+=(std::chrono::duration<double> measuredTime) noexcept
      {
        const double t = measuredTime.count();
        totalTime += t;
        minTime = std::min(minTime, t);
        maxTime = std::max(maxTime, t);
        calls++;
        return *this;
      }

      //! combine two timing statistics
      constexpr TimingStatistics operator+(const TimingStatistics& other) const noexcept
      {
        return {totalTime+other.totalTime, std::min(minTime,other.minTime), std::max(maxTime,other.maxTime), calls+other.calls};
      }

      //! combine two timing statistics
      constexpr TimingStatistics& operator+=(const TimingStatistics& other) noexcept
      {
        *this = *this + other;
        return *this;
      }

      //! allow reading/writing with cereal
      template<class Archive>
      void serialize(Archive & ar)
      {
        ar( CEREAL_NVP(totalTime),
            CEREAL_NVP(minTime),
            CEREAL_NVP(maxTime),
            CEREAL_NVP(calls) );
      }
    };


    //! Helper type for measuring the runtime of some function (or scope)
    //!
    //! Measures the time between construction an destruction:
    //!
    class ScopedTimer final
    {
      public:
        //! start the time measurement
        //!
        //! @param timings  object where the resulting measurement is record
        //!
        explicit ScopedTimer(TimingStatistics& timings) : 
          timings_(timings)
        {
          start_time = clock::now();
        }

        //! stop the time measurement and record the result
        ~ScopedTimer()
        {
          const auto end_time = clock::now();
          timings_ += (end_time - start_time);
        }

      private:
        //! the clock to use
        using clock = std::chrono::steady_clock;

        //! timing statistics object where the result is recorded
        TimingStatistics& timings_;

        //! point in time when the measurement started
        clock::time_point start_time;
    };

    //! helper type for storing timings per function / scope
    using TimingStatisticsMap = std::unordered_map<internal::ScopeInfo, internal::TimingStatistics, internal::ScopeInfo::Hash>;

    //! allow to combine timing statistics by adding them up...
    inline const TimingStatisticsMap& operator+=(TimingStatisticsMap& a, const TimingStatisticsMap& b)
    {
      for(const auto& [scope, timings]: b)
        a[scope] += timings;
      return a;
    }


#ifdef PITTS_USE_LIKWID_MARKER_API
    //! helper class for likwid regions...
    class ScopedLikwidRegion final
    {
      public:
        //! start the likwid region
        explicit ScopedLikwidRegion(const char* name) :
          name_(name)
        {
#pragma omp parallel
          {
            LIKWID_MARKER_START(name_);
          }
        }

        //! stop the likwid region
        ~ScopedLikwidRegion()
        {
#pragma omp parallel
          {
            LIKWID_MARKER_STOP(name_);
          }
        }

      private:
        //! region name
        const char* name_;
    };
#endif


    //! helper type for a timing result with a name
    struct NamedTiming final
    {
      //! description (function name, parameters, etc)
      std::string name;

      //! timing results
      TimingStatistics timings;
    };

    //! gather timings in a vector, so they can be sorted and printed more easily
    inline auto gatherTimings(const TimingStatisticsMap& map, bool mpiGlobal = true)
    {
      // copy to serializable map (ScopeInfo is read-only)
      std::unordered_map<std::string,TimingStatistics> namedMap;
      for(const auto& [scope, timings]: map)
      {
        std::string fullName;
        if( !std::string_view(scope.type_name()).empty() )
          fullName = scope.type_name() + std::string("::");
        fullName += scope.function_name();

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
    //! container for storing timings with createScopedTimer
    inline internal::TimingStatisticsMap globalTimingStatisticsMap;


    //! Measure the runtime of the curent function or scope
    inline auto createScopedTimer(internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
#ifndef PITTS_USE_LIKWID_MARKER_API
      return internal::ScopedTimer(globalTimingStatisticsMap[scope]);
#else
      return std::tuple<internal::ScopedTimer, internal::ScopedLikwidRegion>{
          globalTimingStatisticsMap[scope],
          scope.function_name()};
#endif
    }


    //! Measure the runtime of the curent function or scope (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(scope);
    }


    //! print nice statistics using globalTimingStatisticsMap
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout, bool mpiGlobal = true)
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


#endif // PITTS_TIMER_HPP
