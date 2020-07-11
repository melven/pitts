/*! @file pitts_chunk_ops_avx512.hpp
* @brief AVX512 implementation of common operations with PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-11
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_OPS_AVX512_HPP
#define PITTS_CHUNK_OPS_AVX512_HPP

// includes
#include <immintrin.h>
#include "pitts_chunk_ops_plain.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // specialization for double for dumb compilers
  template<>
  inline void fmadd<float>(const Chunk<float>& a, const Chunk<float>& b, const Chunk<float>& c, Chunk<float>& d)
  {
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      __m512 di = _mm512_fmadd_ps(ai,bi,ci);
      _mm512_store_ps(&d[16*i],di);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(const Chunk<double>& a, const Chunk<double>& b, const Chunk<double>& c, Chunk<double>& d)
  {
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      __m512d di = _mm512_fmadd_pd(ai,bi,ci);
      _mm512_store_pd(&d[8*i],di);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fmadd<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m512 ai = _mm512_set1_ps(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m512d ai = _mm512_set1_pd(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void mul<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m512 ai = _mm512_set1_ps(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_mul_ps(ai,bi);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void mul<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m512d ai = _mm512_set1_pd(a);
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_mul_pd(ai,bi);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(const Chunk<float>& a, const Chunk<float>& b, const Chunk<float>& c, Chunk<float>& d)
  {
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      __m512 di = _mm512_fnmadd_ps(ai,bi,ci);
      _mm512_store_ps(&d[16*i],di);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(const Chunk<double>& a, const Chunk<double>& b, const Chunk<double>& c, Chunk<double>& d)
  {
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      __m512d di = _mm512_fnmadd_pd(ai,bi,ci);
      _mm512_store_pd(&d[8*i],di);
    }
  }
}


#endif // PITTS_CHUNK_OPS_AVX512_HPP
