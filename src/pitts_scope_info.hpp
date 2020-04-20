/*! @file pitts_scope_info.hpp
* @brief Helper class for storing/getting information about the current position in the code
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-04-15
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_SCOPE_INFO_HPP
#define PITTS_SCOPE_INFO_HPP

// includes
#include <experimental/source_location>
#include <array>
#include <string_view>
#include <cstdint>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {

    //! Simple, constexpr hash function for strings (because std::hash is not constexpr!)
    //!
    //! This is known as the djb hash function by Daniel J. Bernstein.
    //!
    constexpr std::uint32_t djb_hash(const std::string_view& str)
    {
      std::uint32_t hash = 5381;
      for(std::uint8_t c: str)
        hash = ((hash << 5) + hash) ^ c;
      return hash;
    }


    //! Helper type to obtain and store the name of the current function / source file
    struct ScopeInfo final : private std::experimental::source_location
    {
      //! constructor that obtains the location of the caller (when called without arguments!)
      constexpr ScopeInfo(std::experimental::source_location here = std::experimental::source_location::current()) : std::experimental::source_location(here) {}

      //! get the name of the enclosing function
      using std::experimental::source_location::function_name;

      //! get the name of the source file
      using std::experimental::source_location::file_name;

      //! get the line number in the source file
      using std::experimental::source_location::line;
    };


    //! Helper type to store information on function/scope arguments
    //!
    //! mostly required for data dimensions, therefore we just support int argument values for now
    //!
    //! @tparam N number of arguments
    //!
    template<std::size_t N = 0>
    struct ArgumentInfo final
    {
      //! argument names
      std::array<const char*, N> names;

      //! argument values
      std::array<int, N> values;

      //! helper function to make a readable string
      std::string to_string() const
      {
        std::string result;
        for(int i = 0; i < N; i++)
        {
          if( i > 0 )
            result += ", ";
          result += names[i];
          result += ": ";
          result += std::to_string(values[i]);
        }
        return result;
      }
    };

  }
}


#endif // PITTS_SCOPE_INFO_HPP
