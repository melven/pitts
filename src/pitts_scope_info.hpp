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
    //! @param str    the string to hash
    //! @param hash   initial hash value, can be used to combine a hash for multiple strings
    //!
    constexpr std::uint32_t djb_hash(const std::string_view& str, std::uint32_t hash = 5381)
    {
      for(std::uint8_t c: str)
        hash = ((hash << 5) + hash) ^ c;
      return hash;
    }


    //! Helper type to obtain and store the name of the current function / source file
    struct ScopeInfo final : private std::experimental::source_location
    {
      //! constructor that obtains the location of the caller (when called without arguments!)
      explicit constexpr ScopeInfo(std::experimental::source_location here = std::experimental::source_location::current()) : 
        std::experimental::source_location(here)
      {
      }

      //! template constructor that also sets the template type
      template<typename T>
      explicit constexpr ScopeInfo(const T&, std::experimental::source_location here = std::experimental::source_location::current()) : 
        std::experimental::source_location(here),
        type_name_(std::experimental::source_location::current().function_name()+9) // 9 == length of "ScopeInfo"
      {
      }

      //! get the name of the enclosing function
      using std::experimental::source_location::function_name;

      //! get the name of the source file
      using std::experimental::source_location::file_name;

      //! get the line number in the source file
      using std::experimental::source_location::line;

      //! get the user-defined type that was set in the constructor
      constexpr std::string_view type_name() const noexcept {return type_name_;}

      //! get a hash (constexpr, required as std::hash is not constexpr)
      constexpr std::uint32_t hash() const noexcept {return hash_;}

    private:
      //! store type name string address
      const char* type_name_ = "";

      //! function_name hash (constexpr)
      std::uint32_t hash_ = djb_hash(function_name(),djb_hash(file_name(),djb_hash(type_name())));
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
