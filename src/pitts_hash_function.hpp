/*! @file pitts_hash_function.hpp
* @brief Helper function for calculting a hash value at compile time
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-12-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_hash_function.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_HASH_FUNCTION)
import pitts_hash_function;
#define PITTS_HASH_FUNCTION_HPP
#endif

// include guard
#ifndef PITTS_HASH_FUNCTION_HPP
#define PITTS_HASH_FUNCTION_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <string_view>
#include <cstdint>

// module export
#ifdef PITTS_USE_MODULES
export module pitts_hash_function;
# define PITTS_MODULE_EXPORT export
#else
# define PITTS_MODULE_EXPORT
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! resulting hash type for djb_hash
    using djb_hash_type = std::uint32_t;

    //! initialization value for djb_hash
    constexpr djb_hash_type djb_hash_init = 5381;

    //! Simple, constexpr hash function for strings (because std::hash is not constexpr!)
    //!
    //! This is known as the djb hash function by Daniel J. Bernstein.
    //!
    //! @param str    the string to hash
    //! @param hash   initial hash value, can be used to combine a hash for multiple strings
    //!
    constexpr djb_hash_type djb_hash(const std::string_view& str, djb_hash_type hash = djb_hash_init) noexcept
    {
      for(std::uint8_t c: str)
        hash = ((hash << 5) + hash) ^ c;
      return hash;
    }

  }
}


#endif // PITTS_HASH_FUNCTION_HPP
