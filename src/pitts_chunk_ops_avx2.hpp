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
