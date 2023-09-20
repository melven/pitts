/*! @file pitts_chunk_ops_avx2.hpp
* @brief AVX2 implementation of common operations with PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-11
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_OPS_AVX2_HPP
#define PITTS_CHUNK_OPS_AVX2_HPP

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
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      __m256 di = _mm256_fmadd_ps(ai,bi,ci);
      _mm256_store_ps(&d[8*i],di);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(const Chunk<double>& a, const Chunk<double>& b, const Chunk<double>& c, Chunk<double>& d)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      __m256d di = _mm256_fmadd_pd(ai,bi,ci);
      _mm256_store_pd(&d[4*i],di);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fmadd_ps(ai,bi,ci);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fmadd<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m256 ai = _mm256_set1_ps(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fmadd_ps(ai,bi,ci);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m256d ai = _mm256_set1_pd(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void mul<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m256 ai = _mm256_set1_ps(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_mul_ps(ai,bi);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void mul<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m256d ai = _mm256_set1_pd(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_mul_pd(ai,bi);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void mul<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_mul_ps(ai,bi);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void mul<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_mul_pd(ai,bi);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(const Chunk<float>& a, const Chunk<float>& b, const Chunk<float>& c, Chunk<float>& d)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      __m256 di = _mm256_fnmadd_ps(ai,bi,ci);
      _mm256_store_ps(&d[8*i],di);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(const Chunk<double>& a, const Chunk<double>& b, const Chunk<double>& c, Chunk<double>& d)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      __m256d di = _mm256_fnmadd_pd(ai,bi,ci);
      _mm256_store_pd(&d[4*i],di);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 ai = _mm256_load_ps(&a[8*i]);
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fnmadd_ps(ai,bi,ci);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d ai = _mm256_load_pd(&a[4*i]);
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fnmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m256 ai = _mm256_set1_ps(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 bi = _mm256_load_ps(&b[8*i]);
      __m256 ci = _mm256_load_ps(&c[8*i]);
      ci = _mm256_fnmadd_ps(ai,bi,ci);
      _mm256_store_ps(&c[8*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m256d ai = _mm256_set1_pd(a);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d bi = _mm256_load_pd(&b[4*i]);
      __m256d ci = _mm256_load_pd(&c[4*i]);
      ci = _mm256_fnmadd_pd(ai,bi,ci);
      _mm256_store_pd(&c[4*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void bcast_sum<float>(Chunk<float>& v)
  {
    static_assert(Chunk<float>::size == 32);
    __m256 v0l = _mm256_add_ps(_mm256_load_ps(&v[0]), _mm256_load_ps(&v[8]));
    __m256 v0h = _mm256_add_ps(_mm256_load_ps(&v[16]), _mm256_load_ps(&v[24]));

    __m256 v1l = _mm256_permute_ps(v0l, 1<<1 | 1<<2 | 1<<3 | 1<<4 );
    __m256 v1h = _mm256_permute_ps(v0h, 1<<1 | 1<<2 | 1<<3 | 1<<4 );
    __m256 v2l = _mm256_add_ps(v0l, v1l);
    __m256 v2h = _mm256_add_ps(v0h, v1h);

    __m256 v3l = _mm256_permute_ps(v2l, 1<<0 | 0<<2 | 3<<4 | 2<<6 );
    __m256 v3h = _mm256_permute_ps(v2h, 1<<0 | 0<<2 | 3<<4 | 2<<6 );
    __m256 v4l = _mm256_add_ps(v2l, v3l);
    __m256 v4h = _mm256_add_ps(v2h, v3h);

    __m256 v5 = _mm256_add_ps(v4h, v4l);

    __m256 v6 = _mm256_permutevar8x32_ps(v5, _mm256_set_epi32(3,2,1,0,7,6,5,4));
    __m256 v7 = _mm256_add_ps(v5, v6);

    _mm256_store_ps(&v[0], v7);
    _mm256_store_ps(&v[8], v7);
    _mm256_store_ps(&v[16], v7);
    _mm256_store_ps(&v[24], v7);
  }

  // specialization for double for dumb compilers
  template<>
  inline void bcast_sum<double>(Chunk<double>& v)
  {
    static_assert(Chunk<double>::size == 16);
    __m256d v0l = _mm256_add_pd(_mm256_load_pd(&v[0]), _mm256_load_pd(&v[4]));
    __m256d v0h = _mm256_add_pd(_mm256_load_pd(&v[8]), _mm256_load_pd(&v[12]));
    __m256d v0 = _mm256_add_pd(v0l, v0h);

    __m256d v1 = _mm256_permute_pd(v0, 1<<0 | 0<<1 | 1<<2 | 0<<3 );
    __m256d v2 = _mm256_add_pd(v0, v1);

    __m256d v3 = _mm256_permute4x64_pd(v2, 2<<0 | 3<<2 | 0<<4 | 1<<6);
    __m256d v4 = _mm256_add_pd(v2, v3);

    _mm256_store_pd(&v[0], v4);
    _mm256_store_pd(&v[4], v4);
    _mm256_store_pd(&v[8], v4);
    _mm256_store_pd(&v[12], v4);
  }
  
  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<float>(const Chunk<float>& src, short index, float value, Chunk<float>& result)
  {
    __m256 val = _mm256_set1_ps(value);
    __m256i ii = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    __m256i eight = _mm256_set1_epi32(8);
    __m256i vindex = _mm256_set1_epi32(index);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256 mask = (__m256)_mm256_cmpeq_epi32(ii, vindex);
      __m256 xi = _mm256_load_ps(&src[8*i]);
      __m256 yi = _mm256_blendv_ps(xi, val, mask);
      _mm256_store_ps(&result[8*i], yi);
      ii = _mm256_add_epi32(ii, eight);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<double>(const Chunk<double>& src, short index, double value, Chunk<double>& result)
  {
    __m256d val = _mm256_set1_pd(value);
    __m256i ii = _mm256_setr_epi64x(0,1,2,3);
    __m256i four = _mm256_set1_epi64x(4);
    __m256i vindex = _mm256_set1_epi64x(index);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256d mask = (__m256d)_mm256_cmpeq_epi64(ii, vindex);
      __m256d xi = _mm256_load_pd(&src[4*i]);
      __m256d yi = _mm256_blendv_pd(xi, val, mask);
      _mm256_store_pd(&result[4*i], yi);
      ii = _mm256_add_epi64(ii, four);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<float>(const Chunk<float>& src, short index, Chunk<float>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    __m256i ii = _mm256_setr_epi32(1,2,3,4,5,6,7,8);
    __m256i vindex = _mm256_set1_epi32(index);
    __m256i eight = _mm256_set1_epi32(8);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256i mask = _mm256_cmpgt_epi32(ii, vindex);
      __m256 vi = _mm256_maskload_ps(&src[8*i], mask);
      _mm256_store_ps(&result[8*i], vi);
      ii = _mm256_add_epi32(ii, eight);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<double>(const Chunk<double>& src, short index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    __m256i ii = _mm256_setr_epi64x(1,2,3,4);
    __m256i vindex = _mm256_set1_epi64x(index);
    __m256i four = _mm256_set1_epi64x(4);

    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256i mask = _mm256_cmpgt_epi64(ii, vindex);
      __m256d vi = _mm256_maskload_pd(&src[4*i], mask);
      _mm256_store_pd(&result[4*i], vi);
      ii = _mm256_add_epi32(ii, four);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<float>(const Chunk<float>& src, short index, Chunk<float>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    __m256i ii = _mm256_setr_epi32(1,2,3,4,5,6,7,8);
    __m256i vindex = _mm256_set1_epi32(index);
    __m256i eight = _mm256_set1_epi32(8);
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256i mask = _mm256_cmpgt_epi32(ii, vindex);
      __m256 vi = _mm256_load_ps(&src[8*i]);
      _mm256_maskstore_ps(&result[8*i], mask, vi);
      ii = _mm256_add_epi32(ii, eight);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<double>(const Chunk<double>& src, short index, Chunk<double>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    __m256i ii = _mm256_setr_epi64x(1,2,3,4);
    __m256i vindex = _mm256_set1_epi64x(index);
    __m256i four = _mm256_set1_epi64x(4);

    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      __m256i mask = _mm256_cmpgt_epi64(ii, vindex);
      __m256d vi = _mm256_load_pd(&src[4*i]);
      _mm256_maskstore_pd(&result[4*i], mask, vi);
      ii = _mm256_add_epi32(ii, four);
    }
  }

  // unaligned load
  template<>
  inline void unaligned_load<float>(const float* src, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_store_ps(&result[8*i], _mm256_loadu_ps(&src[8*i]));
    }
  }

  // unaligned load
  template<>
  inline void unaligned_load<double>(const double* src, Chunk<double>& result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_store_pd(&result[4*i], _mm256_loadu_pd(&src[4*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<float>(const Chunk<float>& src, float* result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_storeu_ps(&result[8*i], _mm256_load_ps(&src[8*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<double>(const Chunk<double>& src, double* result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_storeu_pd(&result[4*i], _mm256_load_pd(&src[4*i]));
    }
  }

  // streaming store
  template<>
  inline void streaming_store<float>(const Chunk<float>& src, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_stream_ps(&result[8*i], _mm256_load_ps(&src[8*i]));
    }
  }

  // streaming store
  template<>
  inline void streaming_store<double>(const Chunk<double>& src, Chunk<double>& result)
  {
    for(short i = 0; i < ALIGNMENT/32; i++)
    {
      _mm256_stream_pd(&result[4*i], _mm256_load_pd(&src[4*i]));
    }
  }
}


#endif // PITTS_CHUNK_OPS_AVX2_HPP
