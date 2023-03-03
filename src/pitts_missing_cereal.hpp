/*! @file pitts_missing_cereal.hpp
 * @brief Workaround for not including cereal everywhere
 * @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
 * @date 2023-03-03
 * @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
 *
 **/

// include guard
#ifndef PITTS_MISSING_CEREAL_HPP
#define PITTS_MISSING_CEREAL_HPP


// delay error to template instantiation
#ifndef CEREAL_NVP

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    // small helper type that defers static_assert evaluation to template instantiation
    // (copied from https://www.fluentcpp.com/2019/08/23/how-to-make-sfinae-pretty-and-robust/)
    template<typename>
    inline constexpr bool cereal_dependent_false_v{ false };

    template<typename T>
    int missing_cereal_include()
    {
      static_assert( cereal_dependent_false_v<T>, "You must include <cereal/cereal.hpp> first!");
      return 0;
    }
  }
}

#define CEREAL_NVP(T)  PITTS::internal::missing_cereal_include<decltype(T)>()

#endif


#endif // PITTS_MISSING_CEREAL_HPP

