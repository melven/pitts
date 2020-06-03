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
#include <chrono>
#include <limits>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include "pitts_scope_info.hpp"


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
    inline TimingStatisticsMap operator+(const TimingStatisticsMap& a, const TimingStatisticsMap& b)
    {
      TimingStatisticsMap ab = a;
      for(const auto& [scope, timings]: b)
        ab[scope] += timings;
      return ab;
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
      return internal::ScopedTimer(globalTimingStatisticsMap[scope]);
    }


    //! Measure the runtime of the curent function or scope (variant with template type information)
    template<typename T>
    inline auto createScopedTimer(internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return createScopedTimer(scope);
    }


    //! print nice statistics using globalTimingStatisticsMap
    inline void printStatistics(bool clear = true, std::ostream& out = std::cout)
    {
      // For sorting and nicer formatting, first copy all stuff into an array of small helper structs
      struct Line final
      {
        std::string name;
        double totalTime;
        std::size_t calls;
      };
      std::vector<Line> lines;
      lines.reserve(globalTimingStatisticsMap.size());

      for(const auto& [scope, timings]: globalTimingStatisticsMap)
      {
        std::string fullName;
        if( !std::string_view(scope.type_name()).empty() )
          fullName = scope.type_name() + std::string("::");
        fullName += scope.function_name();

        lines.emplace_back(Line{fullName, timings.totalTime, timings.calls});
      }
      // sort by decreasing time
      std::sort(lines.begin(), lines.end(), [](const Line& l1, const Line& l2){return l1.totalTime > l2.totalTime;});
      // get maximal length of the name string
      const auto maxNameLen = std::accumulate(lines.begin(), lines.end(), 10, [](std::size_t n, const Line& l){return std::max(n, l.name.size());});

      // actual output
      out << "Timing statistics:\n";
      out << std::left;
      out << std::setw(maxNameLen) << "function" << "\t "
          << std::setw(10) << "time [s]" << "\t "
          << std::setw(10) << "#calls" << "\n";
      for(const auto& line: lines)
      {
        out << std::setw(maxNameLen) << line.name << "\t "
            << std::setw(10) << line.totalTime << "\t "
            << std::setw(10) << line.calls << "\n";
      }

      if( clear )
        globalTimingStatisticsMap.clear();
    }
  }
}


#endif // PITTS_TIMER_HPP
