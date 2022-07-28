/*! @file pitts_type_name.hpp
* @brief Helper functionality to get the name (as a string) of a type (template argument)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-12-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TYPE_NAME_HPP
#define PITTS_TYPE_NAME_HPP

// includes
#include <source_location>
#include <string_view>


//! dummy namespace for PITTS for obtaining compile-time names
namespace PITTS_internal
{
    //! helper function, returns something like "list_of_namespaces::wrapped_type_name<Type>()" and we want to extract "Type"
    template<typename Type>
    static consteval std::string_view wrapped_type_name()
    {
      return std::source_location::current().function_name();
    }
}

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Helper type to obtain the name (as string) of a type
    //!
    //! Example: `TypeName::name<const int>()` returns "const int"
    //!
    struct TypeName final
    {
      public:
        //! get the provided template type as string
        //!
        //! @tparam desired template type / class with cv classifiers
        //! @return string representation of Type
        //!
        template<typename Type>
        static consteval std::string_view name()
        {
          auto wrapped = PITTS_internal::wrapped_type_name<Type>();
          return {wrapped.begin()+prefix_size, wrapped.end()-suffix_size};
        }

      private:
        //! dummy type to calculate offsets in the string returned by wrapper_name
        class ProbeType;

        //! string representation of ProbeType
        static constexpr std::string_view probe_type_name{"PITTS::internal::TypeName::ProbeType"};

        //! compiler-generated "wrapped" string of a template function containing probe_type_name
        static constexpr auto wrapped_probe_type_name = PITTS_internal::wrapped_type_name<ProbeType>();

        //! offset of probe_type_name in wrapped_probe_type_name
        static constexpr auto prefix_size = wrapped_probe_type_name.find(probe_type_name);

        //! offset of the end of probe_type_name in wrapped_probe_type_name
        static constexpr auto suffix_size = wrapped_probe_type_name.length() - prefix_size - probe_type_name.length();

    };
  }
}


#endif // PITTS_TYPE_NAME_HPP
