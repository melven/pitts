// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_chunk.hpp
* @brief Defines PITTS::Chunk SIMD helper type
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
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
#ifdef __AVX512F__
  constexpr auto ALIGNMENT = 128;
#else
  constexpr auto ALIGNMENT = 64;
#endif


  //! helper type for SIMD: a small aligned array of data
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  template<typename T>
  struct alignas(ALIGNMENT) Chunk final : public std::array<T,ALIGNMENT/sizeof(T)>
  {
    static constexpr long long size = ALIGNMENT/sizeof(T);
  };

  //! namespace for helper functionality
  namespace internal
  {
    //! helper function to calculate desired padded length (in #chunks) to avoid cache thrashing (strides of 2^n)
    static constexpr auto paddedChunks(long long nChunks) noexcept
    {
      // pad to next number divisible by PD/2 but not by PD
      constexpr auto PD = 8;
      if( nChunks < 2*PD )
        return nChunks;
      return nChunks + (PD + PD/2 - nChunks % PD) % PD;
    }
  }
}


#endif // PITTS_CHUNK_HPP
