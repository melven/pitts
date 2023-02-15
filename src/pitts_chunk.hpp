/*! @file pitts_chunk.hpp
* @brief Defines PITTS::Chunk SIMD helper type
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_chunk.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_CHUNK)
import pitts_chunk;
#define PITTS_CHUNK_HPP
#endif

// include guard
#ifndef PITTS_CHUNK_HPP
#define PITTS_CHUNK_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <array>

// module export
#ifdef PITTS_USE_MODULES
export module pitts_chunk;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! global alignment (in bytes) to allow SIMD / improve memory accesses
  constexpr auto ALIGNMENT = 128;


  //! helper type for SIMD: a small aligned array of data
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  template<typename T>
  struct alignas(ALIGNMENT) Chunk final : public std::array<T,ALIGNMENT/sizeof(T)>
  {
    static constexpr auto size = ALIGNMENT/sizeof(T);
  };


  // explicit template instantiations
  template struct Chunk<float>;
  template struct Chunk<double>;
}


#endif // PITTS_CHUNK_HPP
