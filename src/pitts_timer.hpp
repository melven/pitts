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
#include "pitts_scope_info.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for runtime measurement data and helper functions
  namespace timing
  {
    //! Helper type for measuring the runtime of some function (or scope)
    //!
    //! Measures the time between construction an destruction:
    //! so it's usually sufficient to just instantiate this class in the beginning of a function.
    //!
    class ScopedTimer final
    {
      public:
        //! constructor: start the time measurement
        ScopedTimer(internal::ScopeInfo scope_ = internal::ScopeInfo()) : scope(scope_), start_time(clock::now()) {}

        //! destructor: stop the time measurement
        ~ScopedTimer()
        {
          const auto end_time = clock::now();
          const std::chrono::duration<double> diff = end_time - start_time;
          std::cout << "Timing " << scope.function_name() << ": " << diff.count() << std::endl;
        }

      private:
        using clock = std::chrono::steady_clock;

        //! where does this measure (e.g. which function)
        internal::ScopeInfo scope;

        //! point in time when the measurement started
        clock::time_point start_time;
    };
  }
}


#endif // PITTS_TIMER_HPP
