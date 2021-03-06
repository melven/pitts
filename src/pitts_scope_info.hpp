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
#include "pitts_hash_function.hpp"
#include "pitts_type_name.hpp"
#include <source_location>
#include <array>
#include <string_view>
#include <cstdint>
#include <bit>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Helper type to obtain and store the name of the current function / source file
    struct ScopeInfo final : private std::source_location
    {
      //! get information on the current scope
      //!
      //! Needs to be called without any argument, so the source_location is evaluated in the context of the caller!
      //!
      static constexpr ScopeInfo current(std::source_location here = std::source_location::current()) noexcept
      {
        return ScopeInfo{here, ""};
      }

      //! get information on the current scope (with additional template type argument)
      //!
      //! Needs to be called without any argument, so the source_location is evaluated in the context of the caller!
      //!
      //! @tparam T  additional template type information for the user
      //!
      template<typename T>
      static constexpr ScopeInfo current(std::source_location here = std::source_location::current()) noexcept
      {
        return ScopeInfo{here, TypeName::name<T>()};
      }

      //! get the name of the enclosing function
      using std::source_location::function_name;

      //! get the name of the source file
      using std::source_location::file_name;

      //! get the line number in the source file
      using std::source_location::line;

      //! get the user-defined type that was set in the constructor
      constexpr std::string_view type_name() const noexcept {return type_name_;}

      //! Callable to get hash from ScopeInfo object, can be used with std::unordered_map
      struct Hash final
      {
        constexpr auto operator()(const ScopeInfo& info) const noexcept
        {
          return info.hash_;
        }
      };

      //! comparison operator for std::unordered_map
      constexpr bool operator==(const ScopeInfo& other) const noexcept
      {
        return function_name() == other.function_name() && file_name() == other.file_name() && type_name() == other.type_name();
      }

    private:
      //! store type name string address
      std::string_view type_name_;

      //! function_name hash (constexpr, required as std::hash is not constexpr)
      djb_hash_type hash_ = djb_hash(function_name(),djb_hash(file_name(),djb_hash(type_name())));

      //! internal constructor, call current instead!
      constexpr explicit ScopeInfo(std::source_location where, std::string_view typeStr) : std::source_location(where), type_name_(typeStr) {}
    };


    //! Helper type to store information on function/scope arguments
    //!
    //! mostly required for data dimensions, therefore we just support integer argument values for now
    //!
    //! @tparam N number of arguments
    //!
    template<std::size_t N = 0>
    struct ArgumentInfo final
    {
      //! argument names
      std::array<const char*, N> names;

      //! argument values
      std::array<long long, N> values;

      //! make a readable string
      std::string to_string() const
      {
        std::string result;
        for(std::size_t i = 0; i < N; i++)
        {
          if( !names[i] )
            break;
          if( i > 0 )
            result += ", ";
          result += names[i];
          result += ": ";
          result += std::to_string(values[i]);
        }
        return result;
      }

      //! calculate a hash of the argument values (constexpr)
      constexpr djb_hash_type hash_values(djb_hash_type hash = djb_hash_init) const noexcept
      {
        constexpr auto size = sizeof(values[0]) * N;
        using byte_array = std::array<char, size>;
        const auto raw_buff = std::bit_cast<byte_array, decltype(values)>(values);
        const auto raw_view = std::string_view{raw_buff.begin(), raw_buff.end()};
        return djb_hash(raw_view, hash);
      }
    };


    //! ArgumentInfo with fixed (maximal) number of arguments to allow non-template struct
    using FixedArgumentInfo = ArgumentInfo<5>;


    //! helper type for combining scope and argument information
    //!
    //! For comparison and hashing only the actual argument values are considered!
    //!
    struct ScopeWithArgumentInfo final
    {
      //! the encompassing scope
      ScopeInfo scope;

      //! the (currently) provided argument(s)
      FixedArgumentInfo args;

      //! number of measurements ("sub-scopes") per function,
      //!
      //! This is used to accumulate function-level call counts while measuring parts of a function.
      //!
      double callsPerFunction = 1;

      //! Callable to combine hashes from ScopeInfo and ArgumentInfo objects, can be used with std::unordered_map
      struct Hash final
      {
        constexpr auto operator()(const ScopeWithArgumentInfo& info) const noexcept
        {
          ScopeInfo::Hash helper;
          return info.args.hash_values(helper(info.scope));
        }
      };

      //! comparison operator for std::unordered_map
      constexpr bool operator==(const ScopeWithArgumentInfo& other) const noexcept
      {
        return scope == other.scope && args.values == other.args.values;
      }
    };
  }
}


#endif // PITTS_SCOPE_INFO_HPP
