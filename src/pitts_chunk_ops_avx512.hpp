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

  // specialization for double for dumb compilers
  template<>
  inline double sum<double>(const Chunk<double>& v)
  {
    // not sure if this is actually faster that vadd;
    // I assume it pipelines better with other AVX512 ops
    __m512d v00 = _mm512_load_pd(&v[0]);
    __m512d v80 = _mm512_load_pd(&v[8]);

    __m512d v01 = _mm512_permute_pd(v00, 1<<0 | 1<<2 | 1<<4 | 1<<6 );
    __m512d v81 = _mm512_permute_pd(v80, 1<<0 | 1<<2 | 1<<4 | 1<<6 );
    __m512d v02 = _mm512_add_pd(v00, v01);
    __m512d v82 = _mm512_add_pd(v80, v81);

    __m512d v03 = _mm512_permutex_pd(v02, 2 | 3<<2 | 0<<4 | 1<<6);
    __m512d v83 = _mm512_permutex_pd(v82, 2 | 3<<2 | 0<<4 | 1<<6);
    __m512d v04 = _mm512_add_pd(v02, v03);
    __m512d v84 = _mm512_add_pd(v82, v83);

    __m512d v05 = _mm512_permutexvar_pd(_mm512_set_epi64(3,2,1,0,7,6,5,4), v04);
    __m512d v85 = _mm512_permutexvar_pd(_mm512_set_epi64(3,2,1,0,7,6,5,4), v84);
    __m512d v06 = _mm512_add_pd(v04, v05);
    __m512d v86 = _mm512_add_pd(v84, v85);

    return v06[0] + v86[0];
  }

  // specialization for double for dumb compilers
  template<>
  inline void bcast_sum<double>(Chunk<double>& v)
  {
    __m512d v0 = _mm512_add_pd(_mm512_load_pd(&v[0]), _mm512_load_pd(&v[8]));

    __m512d v1 = _mm512_permute_pd(v0, 1<<0 | 1<<2 | 1<<4 | 1<<6 );
    __m512d v2 = _mm512_add_pd(v0, v1);

    __m512d v3 = _mm512_permutex_pd(v2, 2 | 3<<2 | 0<<4 | 1<<6);
    __m512d v4 = _mm512_add_pd(v2, v3);

    __m512d v5 = _mm512_permutexvar_pd(_mm512_set_epi64(3,2,1,0,7,6,5,4), v4);
    __m512d v6 = _mm512_add_pd(v4, v5);

    _mm512_store_pd(&v[0], v6);
    _mm512_store_pd(&v[8], v6);
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<double>(const Chunk<double>& src, int index, double value, Chunk<double>& result)
  {
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      __mmask8 mask = (1<<index)>>(8*i);
      __m512d xi = _mm512_load_pd(&src[8*i]);
      __m512d yi = _mm512_mask_broadcastsd_pd(xi, mask, _mm_set_pd(0,value));
      _mm512_store_pd(&result[8*i], yi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<double>(const Chunk<double>& src, int index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<index)>>(8*i);
      __m512d vi = _mm512_maskz_load_pd(mask, &src[8*i]);
      _mm512_store_pd(&result[8*i], vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<double>(const Chunk<double>& src, int index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(int i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<index)>>(8*i);
      __m512d vi = _mm512_load_pd(&src[8*i]);
      _mm512_mask_store_pd(&result[8*i], mask, vi);
    }
  }
}


#endif // PITTS_CHUNK_OPS_AVX512_HPP
