/*! @file pitts_chunk_ops.hpp
* @brief helper functions for common SIMD operations using PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-30
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_OPS_HPP
#define PITTS_CHUNK_OPS_HPP

// includes
#include <immintrin.h>
#include "pitts_chunk.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! add the element-wise product of two chunks
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] += a[i]*b[i];
  }

  //! add the element-wise product of a scalar and a chunk
  template<typename T>
  inline void fmadd(T a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] += a*b[i];
  }

  //! add the negative element-wise product of two chunks
  template<typename T>
  inline void fnmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      c[i] -= a[i]*b[i];
  }

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

  //! small helper function to extract the negative sign of the elements in a chunk
  template<typename T>
  inline void negative_sign(const Chunk<T>& v, Chunk<T>& nsgn)
  {
    for(int i = 0; i < Chunk<T>::size; i++)
      nsgn[i] = v[i] == T(0) ? T(-1) : -v[i] / std::abs(v[i]);
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

#ifdef __AVX2__
  // specialization for float for dumb compilers
  template<>
  inline void fmadd<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
#if defined(__AVX512F__)
    __m512 ai = _mm512_set1_ps(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
#elif defined(__AVX2__)
    __m256 ai = _mm256_set1_ps(a);
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
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

#ifdef __AVX2__
  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
#if defined(__AVX512F__)
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fnmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
#elif defined(__AVX2__)
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fnmadd_ps(ai,bi,ci);
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
  inline void fnmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
#if defined(__AVX512F__)
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fnmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
#elif defined(__AVX2__)
    for(int i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fnmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
#else
#error "PITTS requires at least AVX2 support!"
#endif
  }
#endif

#if defined(__AVX512F__)
  // specialization for double for dumb compilers
  template<>
  inline void negative_sign<double>(const Chunk<double>& v, Chunk<double>& nsgn)
  {
    __mmask8 mask[ALIGNMENT/64];
    const auto zero = _mm512_setzero_pd();
    const auto one = _mm512_set1_pd(1);
    const auto minus_one = _mm512_set1_pd(-1);
    for(int i = 0; i < ALIGNMENT/64; i++)
      mask[i] = _mm512_cmpnlt_pd_mask(_mm512_load_pd(&(v[i*8])), zero);
    for(int i = 0; i < ALIGNMENT/64; i++)
      _mm512_store_pd(&(nsgn[i*8]), _mm512_mask_blend_pd(mask[i], one, minus_one));
  }
#endif

}


#endif // PITTS_CHUNK_OPS_HPP
