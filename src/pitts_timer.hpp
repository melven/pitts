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
      const TimingStatistics& operator+=(std::chrono::duration<double> measuredTime)
      {
        const double t = measuredTime.count();
        totalTime += t;
        minTime = std::min(minTime, t);
        maxTime = std::max(maxTime, t);
        calls++;
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
  }


  //! namespace for runtime measurement data and helper functions
  namespace timing
  {
    internal::TimingStatisticsMap globalTimingStatisticsMap;

    //! Measure the runtime of the curent function or scope
    auto createScopedTimer(internal::ScopeInfo scope = internal::ScopeInfo::current())
    {
      return internal::ScopedTimer(globalTimingStatisticsMap[scope]);
    }


    //! Measure the runtime of the curent function or scope (variant with template type information)
    template<typename T>
    auto createScopedTimer(internal::ScopeInfo scope = internal::ScopeInfo::template current<T>())
    {
      return internal::ScopedTimer(globalTimingStatisticsMap[scope]);
    }
  }
}


#endif // PITTS_TIMER_HPP
