/*! @file pitts_random.hpp
* @brief Helper functionality for generating random numbers
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_RANDOM_HPP
#define PITTS_RANDOM_HPP

// includes
#include <random>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! global random number generator in PITTS, seeded in PITTS::initialize
    inline std::mt19937_64 randomGenerator;

    //! helper function to generate a random seed
    //!
    //! @warning Even though this is the usual way to initialize an RNG, this is most-probably biased:
    //!          The internal state of the RNG is much bigger than the generated (64) bytes...
    //!
    inline std::uint_fast64_t generateRandomSeed()
    {
        std::random_device randomDevice;
        std::uint_fast64_t result = randomDevice();
        result = (result << 32) | randomDevice();
        return result;
    }
  }
}

#endif // PITTS_RANDOM_HPP