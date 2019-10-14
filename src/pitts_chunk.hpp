/*! @file pitts_chunk.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_HPP
#define PITTS_CHUNK_HPP

// includes
#include <array>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! global alignment (in bytes) to allow SIMD / improve memory accesses
  constexpr auto ALIGNMENT = 512;


  //! helper type for SIMD: a small aligned array of data
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  template<typename T>
  struct alignas(ALIGNMENT) Chunk final : public std::array<T,ALIGNMENT/sizeof(T)>
  {
    static constexpr auto size = ALIGNMENT/sizeof(T);
  };

  //! small helper function to add up the element-wise product of two chunks
  template<typename T>
  constexpr void fmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] += a[i]*b[i];
  }
}


#endif // PITTS_CHUNK_HPP
