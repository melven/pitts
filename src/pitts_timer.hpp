// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_timer.hpp
* @brief Helper functionality for measuring the runtime of C++ functions / scopes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-04-14
*
**/

// include guard
#ifndef PITTS_TIMER_HPP
#define PITTS_TIMER_HPP

// includes
#include "pitts_scope_info.hpp"
#include <chrono>
#include <limits>
#include <unordered_map>
#include <iostream>

#ifdef PITTS_USE_LIKWID_MARKER_API
#include <likwid.h>
#include <cstring>
#endif

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
          // workaround for likwid bug: https://github.com/RRZE-HPC/likwid/issues/551
          std::strncpy(shortenedName_, name_, 90);
          shortenedName_[90] = '\0';
#pragma omp parallel
          {
            LIKWID_MARKER_START(shortenedName_);
          }
        }

        //! stop the likwid region
        ~ScopedLikwidRegion()
        {
#pragma omp parallel
          {
            LIKWID_MARKER_STOP(shortenedName_);
          }
        }

      private:
        //! region name
        const char* name_;
        char shortenedName_[91];
    };
#endif


    //! gather timings in a vector, so they can be sorted and printed more easily
    auto gatherTimings(const TimingStatisticsMap& map, bool mpiGlobal = true);
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
    void printStatistics(bool clear = true, std::ostream& out = std::cout, bool mpiGlobal = true);

    //! helper function to clear statistics
    inline void clearStatistics()
    {
#ifdef PITTS_USE_LIKWID_MARKER_API
      for(auto& it: globalTimingStatisticsMap)
      {
        char shortenedName[91];
        std::strncpy(shortenedName, it.first.function_name(), 90);
        shortenedName[90] = '\0';
#pragma omp parallel
        {
          LIKWID_MARKER_RESET(shortenedName);
        }
      }
#endif
      globalTimingStatisticsMap.clear();
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_timer_impl.hpp"
#endif

#endif // PITTS_TIMER_HPP
