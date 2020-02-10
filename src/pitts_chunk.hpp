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
#include <immintrin.h>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
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

  //! small helper function to add up the element-wise product of two chunks
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] += a[i]*b[i];
  }

#ifdef __AVX2__
  // specialization for double for dumb compilers
  template<>
  inline void fmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
#if defined(__AVX512F__)
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
#elif defined(__AVX2__)
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fmadd_ps(ai,bi,ci);
      _mm256_store_ps(&c[8*i],ci);
    }
#else
#error "PITTS requires at least AVX2 support!"
#endif
  }
#endif

#ifdef __AVX2__
  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
#if defined(__AVX512F__)
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
#elif defined(__AVX2__)
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
#else
#error "PITTS requires at least AVX2 support!"
#endif
  }
#endif

  //! small helper function to add up the element-wise product of two chunks
  template<typename T>
  inline void fmadd(T a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] += a*b[i];
  }

#ifdef __AVX2__
  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
#if defined(__AVX512F__)
    __m512d ai = _mm512_set1_pd(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
#elif defined(__AVX2__)
    __m256d ai = _mm256_set1_pd(a);
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
#else
#error "PITTS requires at least AVX2 support!"
#endif
  }
#endif

  //! small helper function to sum up all elements of a chunk
  template<typename T>
  inline T sum(const Chunk<T>& a)
  {
    T tmp = T(0);
    for(int i = 0; i < Chunk<T>::size; i++)
      tmp += a[i];
    return tmp;
  }

  //! small helper function to scale and sum up all elements of a chunk
  template<typename T>
  inline T scaled_sum(T scale, const Chunk<T>& a)
  {
    T tmp = T(0);
    for(int i = 0; i < Chunk<T>::size; i++)
      tmp += scale * a[i];
    return tmp;
  }
}


#endif // PITTS_CHUNK_HPP
