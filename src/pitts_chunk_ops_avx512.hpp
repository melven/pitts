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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
    for(short i = 0; i < ALIGNMENT/64; i++)
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
  inline float sum<float>(const Chunk<float>& v)
  {
    // not sure if this is actually faster that vadd;
    // I assume it pipelines better with other AVX512 ops
    __m512 vl0 = _mm512_load_ps(&v[0]);
    __m512 vh0 = _mm512_load_ps(&v[16]);

    __m512 vl1 = _mm512_permute_ps(vl0, 1<<0 | 0<<2 | 3<<4 | 2<<6 );
    __m512 vh1 = _mm512_permute_ps(vh0, 1<<0 | 0<<2 | 3<<4 | 2<<6 );
    __m512 vl2 = _mm512_add_ps(vl0, vl1);
    __m512 vh2 = _mm512_add_ps(vh0, vh1);

    __m512 vl3 = _mm512_permute_ps(vl2, 2<<0 | 3<<2 | 0<<4 | 1<<6 );
    __m512 vh3 = _mm512_permute_ps(vh2, 2<<0 | 3<<2 | 0<<4 | 1<<6 );
    __m512 vl4 = _mm512_add_ps(vl2, vl3);
    __m512 vh4 = _mm512_add_ps(vh2, vh3);

    __m512 vl5 = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), vl4);
    __m512 vh5 = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), vh4);
    __m512 vl6 = _mm512_add_ps(vl4, vl5);
    __m512 vh6 = _mm512_add_ps(vh4, vh5);

    __m512 vl7 = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), vl6);
    __m512 vh7 = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), vh6);
    __m512 vl8 = _mm512_add_ps(vl6, vl7);
    __m512 vh8 = _mm512_add_ps(vh6, vh7);

    return vh8[0] + vl8[0];
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
  inline void bcast_sum<float>(Chunk<float>& v)
  {
    __m512 v0 = _mm512_add_ps(_mm512_load_ps(&v[0]), _mm512_load_ps(&v[16]));

    __m512 v1 = _mm512_permute_ps(v0, 1<<0 | 0<<2 | 3<<4 | 2<<6 );
    __m512 v2 = _mm512_add_ps(v0, v1);

    __m512 v3 = _mm512_permute_ps(v2, 2<<0 | 3<<2 | 0<<4 | 1<<6 );
    __m512 v4 = _mm512_add_ps(v2, v3);

    __m512 v5 = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), v4);
    __m512 v6 = _mm512_add_ps(v4, v5);

    __m512 v7 = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), v6);
    __m512 v8 = _mm512_add_ps(v6, v7);

    _mm512_store_ps(&v[0], v8);
    _mm512_store_ps(&v[16], v8);
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
  inline void index_bcast<float>(const Chunk<float>& src, short index, float value, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __mmask16 mask = (1<<index)>>(16*i);
      __m512 xi = _mm512_load_ps(&src[16*i]);
      __m512 yi = _mm512_mask_broadcastss_ps(xi, mask, _mm_set_ps(0,0,0,value));
      _mm512_store_ps(&result[16*i], yi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<double>(const Chunk<double>& src, short index, double value, Chunk<double>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __mmask8 mask = (1<<index)>>(8*i);
      __m512d xi = _mm512_load_pd(&src[8*i]);
      __m512d yi = _mm512_mask_broadcastsd_pd(xi, mask, _mm_set_pd(0,value));
      _mm512_store_pd(&result[8*i], yi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<float>(const Chunk<float>& src, short index, Chunk<float>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask16 mask = (all<<index)>>(16*i);
      __m512 vi = _mm512_maskz_load_ps(mask, &src[16*i]);
      _mm512_store_ps(&result[16*i], vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<double>(const Chunk<double>& src, short index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<index)>>(8*i);
      __m512d vi = _mm512_maskz_load_pd(mask, &src[8*i]);
      _mm512_store_pd(&result[8*i], vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<float>(const Chunk<float>& src, short index, Chunk<float>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask16 mask = (all<<index)>>(16*i);
      __m512 vi = _mm512_load_ps(&src[16*i]);
      _mm512_mask_store_ps(&result[16*i], mask, vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<double>(const Chunk<double>& src, short index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<index)>>(8*i);
      __m512d vi = _mm512_load_pd(&src[8*i]);
      _mm512_mask_store_pd(&result[8*i], mask, vi);
    }
  }

  // unaligned load
  template<>
  inline void unaligned_load<float>(const float* src, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_store_ps(&result[16*i], _mm512_loadu_ps(&src[16*i]));
    }
  }

  // unaligned load
  template<>
  inline void unaligned_load<double>(const double* src, Chunk<double>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_store_pd(&result[8*i], _mm512_loadu_pd(&src[8*i]));
    }
  }


  // streaming store
  template<>
  inline void streaming_store<float>(const Chunk<float>& src, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_stream_ps(&result[16*i], _mm512_load_ps(&src[16*i]));
    }
  }

  // streaming store
  template<>
  inline void streaming_store<double>(const Chunk<double>& src, Chunk<double>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_stream_pd(&result[8*i], _mm512_load_pd(&src[8*i]));
    }
  }
}


#endif // PITTS_CHUNK_OPS_AVX512_HPP
