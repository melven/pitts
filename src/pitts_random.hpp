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
#include <complex>
#include <numeric>
#include <cmath>

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

    //! uniform distribution with abs(x) <= 1
    template<typename T>
    struct UniformUnitDistribution
    {
        //! sample from the distribution using a random generator
        template<std::uniform_random_bit_generator Generator>
        [[nodiscard]] T operator()(Generator& g) const
        {
            std::uniform_real_distribution<T> distribution(-1, 1);
            return distribution(g);
        }

        //! discard n elements from the random generator
        template<std::uniform_random_bit_generator Generator>
        void discard(unsigned long long n, Generator& g) const
        {
            constexpr auto wordsPerNumber = 1 + (sizeof(T)-1) / Generator::word_size;
            g.discard(n*wordsPerNumber);
        }
    };

    //! specialization for complex types
    template<typename T>
    struct UniformUnitDistribution<std::complex<T>>
    {
        //! sample from the distribution using a random generator
        template<std::uniform_random_bit_generator Generator>
        [[nodiscard]] std::complex<T> operator()(Generator& g)
        {
            std::uniform_real_distribution<T> distributionR2(0, 1);
            std::uniform_real_distribution<T> distributionPhi(0, 2*std::numbers::pi);
            const T r = std::sqrt(distributionR2(g));
            const T phi = distributionPhi(g);
            return std::complex<T>(r*std::cos(phi), r*std::sin(phi));
        }

        //! discard n elements from the random generator
        template<std::uniform_random_bit_generator Generator>
        void discard(unsigned long long n, Generator& g) const
        {
            constexpr auto wordsPerNumber = 1 + (sizeof(T)-1) / Generator::word_size;
            g.discard(2*n*wordsPerNumber);
        }
    };
  }
}

#endif // PITTS_RANDOM_HPP