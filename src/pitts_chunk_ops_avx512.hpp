// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_chunk_ops_avx512.hpp
* @brief AVX512 implementation of common operations with PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-11
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

  // specialization for complex float for dumb compilers
  template<>
  inline void fmadd<std::complex<float>>(const Chunk<std::complex<float>>& a, const Chunk<std::complex<float>>& b, const Chunk<std::complex<float>>& c, Chunk<std::complex<float>>& d)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 ai_rr = _mm512_permute_ps(ai, 0<<0 | 0<<2 | 2<<4 | 2<<6);
      __m512 ai_ii = _mm512_permute_ps(ai, 1<<0 | 1<<2 | 3<<4 | 3<<6);
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmaddsub_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_ps(ai_rr, bi, ci);
      _mm512_store_ps(&d[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fmadd<std::complex<double>>(const Chunk<std::complex<double>>& a, const Chunk<std::complex<double>>& b, const Chunk<std::complex<double>>& c, Chunk<std::complex<double>>& d)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d ai_rr = _mm512_permute_pd(ai, 0<<0 | 0<<2 | 0<<4 | 0<<6);
      __m512d ai_ii = _mm512_permute_pd(ai, 3<<0 | 3<<2 | 3<<4 | 3<<6);
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      // ci_r' <- - ci_r + ai_i*bi_i
      // ci_i' <- + ci_i + ai_i*bi_r
      ci = _mm512_fmaddsub_pd(ai_ii, bi_ir, ci);
      // ci_r'' <- - ci_r' + ai_r*bi_r = ci_r - ai_i*bi_i + ai_r*bi_r
      // ci_i'' <- + ci_i' + ai_r*bi_i = ci_i + ai_i*bi_r + ai_r*bi_i
      ci = _mm512_fmaddsub_pd(ai_rr, bi, ci);
      _mm512_store_pd(&d[4*i],ci);
    }
  }


  // specialization for double for dumb compilers
  template<>
  inline void fmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for complex float for dumb compilers
  template<>
  inline void fmadd<std::complex<float>>(const Chunk<std::complex<float>>& a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 ai_rr = _mm512_permute_ps(ai, 0<<0 | 0<<2 | 2<<4 | 2<<6);
      __m512 ai_ii = _mm512_permute_ps(ai, 1<<0 | 1<<2 | 3<<4 | 3<<6);
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmaddsub_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_ps(ai_rr, bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fmadd<std::complex<double>>(const Chunk<std::complex<double>>& a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d ai_rr = _mm512_permute_pd(ai, 0<<0 | 0<<2 | 0<<4 | 0<<6);
      __m512d ai_ii = _mm512_permute_pd(ai, 3<<0 | 3<<2 | 3<<4 | 3<<6);
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      ci = _mm512_fmaddsub_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_pd(ai_rr, bi, ci);
      _mm512_store_pd(&c[4*i],ci);
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

  // specialization for complex float for dumb compilers
  template<>
  inline void fmadd<std::complex<float>>(std::complex<float> a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    __m512 ai_rr = _mm512_set1_ps(a.real());
    __m512 ai_ii = _mm512_set1_ps(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmaddsub_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_ps(ai_rr, bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fmadd<std::complex<double>>(std::complex<double> a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    __m512d ai_rr = _mm512_set1_pd(a.real());
    __m512d ai_ii = _mm512_set1_pd(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      ci = _mm512_fmaddsub_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_pd(ai_rr, bi, ci);
      _mm512_store_pd(&c[4*i],ci);
    }
  }

  // specialization for complex float for dumb compilers
  template<>
  inline Chunk<std::complex<float>> conj<std::complex<float>>(const Chunk<std::complex<float>>& a)
  {
    float neg_zero = std::copysign(float(0), float(-1));
    __m512 sign_mask = _mm512_broadcast_f32x2(_mm_set_ps(0, 0, neg_zero, 0));
    Chunk<std::complex<float>> b;
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_xor_ps(ai, sign_mask);
      _mm512_store_ps(&b[8*i],bi);
    }
    return b;
  }

  // specialization for complex float for dumb compilers
  template<>
  inline Chunk<std::complex<double>> conj<std::complex<double>>(const Chunk<std::complex<double>>& a)
  {
    double neg_zero = std::copysign(double(0), double(-1));
    __m512d sign_mask = _mm512_broadcast_f64x2(_mm_set_pd(neg_zero, 0));
    Chunk<std::complex<double>> b;
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_xor_pd(ai, sign_mask);
      _mm512_store_pd(&b[4*i],bi);
    }
    return b;
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

  // specialization for complex float for dumb compilers
  template<>
  inline void mul<std::complex<float>>(std::complex<float> a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    __m512 ai_rr = _mm512_set1_ps(a.real());
    __m512 ai_ii = _mm512_set1_ps(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_setzero_ps();
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmaddsub_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_ps(ai_rr, bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void mul<std::complex<double>>(std::complex<double> a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    __m512d ai_rr = _mm512_set1_pd(a.real());
    __m512d ai_ii = _mm512_set1_pd(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_setzero_pd();
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      ci = _mm512_fmaddsub_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_pd(ai_rr, bi, ci);
      _mm512_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void mul<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_mul_ps(ai,bi);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void mul<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_mul_pd(ai,bi);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for complex float for dumb compilers
  template<>
  inline void mul<std::complex<float>>(const Chunk<std::complex<float>>& a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_setzero_ps();
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 ai_rr = _mm512_permute_ps(ai, 0<<0 | 0<<2 | 2<<4 | 2<<6);
      __m512 ai_ii = _mm512_permute_ps(ai, 1<<0 | 1<<2 | 3<<4 | 3<<6);
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmaddsub_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_ps(ai_rr, bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void mul<std::complex<double>>(const Chunk<std::complex<double>>& a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_setzero_pd();
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d ai_rr = _mm512_permute_pd(ai, 0<<0 | 0<<2 | 0<<4 | 0<<6);
      __m512d ai_ii = _mm512_permute_pd(ai, 3<<0 | 3<<2 | 3<<4 | 3<<6);
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      ci = _mm512_fmaddsub_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmaddsub_pd(ai_rr, bi, ci);
      _mm512_store_pd(&c[4*i],ci);
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

  // specialization for complex float for dumb compilers
  template<>
  inline void fnmadd<std::complex<float>>(const Chunk<std::complex<float>>& a, const Chunk<std::complex<float>>& b, const Chunk<std::complex<float>>& c, Chunk<std::complex<float>>& d)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 ai_rr = _mm512_permute_ps(ai, 0<<0 | 0<<2 | 2<<4 | 2<<6);
      __m512 ai_ii = _mm512_permute_ps(ai, 1<<0 | 1<<2 | 3<<4 | 3<<6);
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      __m512 neg_bi = -bi;
      ci = _mm512_fmsubadd_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_ps(ai_rr, neg_bi, ci);
      _mm512_store_ps(&d[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fnmadd<std::complex<double>>(const Chunk<std::complex<double>>& a, const Chunk<std::complex<double>>& b, const Chunk<std::complex<double>>& c, Chunk<std::complex<double>>& d)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d ai_rr = _mm512_permute_pd(ai, 0<<0 | 0<<2 | 0<<4 | 0<<6);
      __m512d ai_ii = _mm512_permute_pd(ai, 3<<0 | 3<<2 | 3<<4 | 3<<6);
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      __m512d neg_bi = -bi;
      ci = _mm512_fmsubadd_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_pd(ai_rr, neg_bi, ci);
      _mm512_store_pd(&d[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(const Chunk<float>& a, const Chunk<float>& b, Chunk<float>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[16*i]);
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fnmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(const Chunk<double>& a, const Chunk<double>& b, Chunk<double>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[8*i]);
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fnmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for complex float for dumb compilers
  template<>
  inline void fnmadd<std::complex<float>>(const Chunk<std::complex<float>>& a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 ai = _mm512_load_ps(&a[8*i]);
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 ai_rr = _mm512_permute_ps(ai, 0<<0 | 0<<2 | 2<<4 | 2<<6);
      __m512 ai_ii = _mm512_permute_ps(ai, 1<<0 | 1<<2 | 3<<4 | 3<<6);
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      __m512 neg_bi = -bi;
      ci = _mm512_fmsubadd_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_ps(ai_rr, neg_bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fnmadd<std::complex<double>>(const Chunk<std::complex<double>>& a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d ai = _mm512_load_pd(&a[4*i]);
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d ai_rr = _mm512_permute_pd(ai, 0<<0 | 0<<2 | 0<<4 | 0<<6);
      __m512d ai_ii = _mm512_permute_pd(ai, 3<<0 | 3<<2 | 3<<4 | 3<<6);
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      __m512d neg_bi = -bi;
      ci = _mm512_fmsubadd_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_pd(ai_rr, neg_bi, ci);
      _mm512_store_pd(&c[4*i],ci);
    }
  }

  // specialization for float for dumb compilers
  template<>
  inline void fnmadd<float>(float a, const Chunk<float>& b, Chunk<float>& c)
  {
    __m512 ai = _mm512_set1_ps(a);
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[16*i]);
      __m512 ci = _mm512_load_ps(&c[16*i]);
      ci = _mm512_fnmadd_ps(ai,bi,ci);
      _mm512_store_ps(&c[16*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline void fnmadd<double>(double a, const Chunk<double>& b, Chunk<double>& c)
  {
    __m512d ai = _mm512_set1_pd(a);
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[8*i]);
      __m512d ci = _mm512_load_pd(&c[8*i]);
      ci = _mm512_fnmadd_pd(ai,bi,ci);
      _mm512_store_pd(&c[8*i],ci);
    }
  }

  // specialization for complex float for dumb compilers
  template<>
  inline void fnmadd<std::complex<float>>(std::complex<float> a, const Chunk<std::complex<float>>& b, Chunk<std::complex<float>>& c)
  {
    __m512 ai_rr = _mm512_set1_ps(-a.real());
    __m512 ai_ii = _mm512_set1_ps(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512 bi = _mm512_load_ps(&b[8*i]);
      __m512 ci = _mm512_load_ps(&c[8*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512 bi_ir = _mm512_permute_ps(bi, 1<<0 | 0<<2 | 3<<4 | 2<<6);
      ci = _mm512_fmsubadd_ps(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_ps(ai_rr, bi, ci);
      _mm512_store_ps(&c[8*i],ci);
    }
  }

  // specialization for complex double for dumb compilers
  template<>
  inline void fnmadd<std::complex<double>>(std::complex<double> a, const Chunk<std::complex<double>>& b, Chunk<std::complex<double>>& c)
  {
    __m512d ai_rr = _mm512_set1_pd(-a.real());
    __m512d ai_ii = _mm512_set1_pd(a.imag());
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      __m512d bi = _mm512_load_pd(&b[4*i]);
      __m512d ci = _mm512_load_pd(&c[4*i]);
      // c_r = c_r + a_r*b_r - a_i*b_i
      // c_i = c_i + a_r*b_i + a_i*b_r
      __m512d bi_ir = _mm512_permute_pd(bi, 1<<0 | 1<<2 | 1<<4 | 1<<6);
      ci = _mm512_fmsubadd_pd(ai_ii, bi_ir, ci);
      ci = _mm512_fmsubadd_pd(ai_rr, bi, ci);
      _mm512_store_pd(&c[4*i],ci);
    }
  }

  // specialization for double for dumb compilers
  template<>
  inline float sum<float>(const Chunk<float>& v)
  {
    // TODO
    static_assert(Chunk<float>::size == 32);
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
    static_assert(Chunk<double>::size == 16);
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
  inline std::complex<float> sum<std::complex<float>>(const Chunk<std::complex<float>>& v)
  {
    static_assert(Chunk<std::complex<float>>::size == 16);
    // not sure if this is actually faster that vadd;
    // I assume it pipelines better with other AVX512 ops
    __m512 vl2 = _mm512_load_ps(&v[0]);
    __m512 vh2 = _mm512_load_ps(&v[8]);

    __m256 vl30 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vl2), 0));
    __m256 vh30 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vh2), 0));
    __m256 vl31 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vl2), 1));
    __m256 vh31 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vh2), 1));
    __m256 vl4 = _mm256_add_ps(vl30, vl31);
    __m256 vh4 = _mm256_add_ps(vh30, vh31);

    __m128 vl50 = _mm256_extractf128_ps(vl4, 0);
    __m128 vh50 = _mm256_extractf128_ps(vh4, 0);
    __m128 vl51 = _mm256_extractf128_ps(vl4, 1);
    __m128 vh51 = _mm256_extractf128_ps(vh4, 1);
    __m128 vl6 = _mm_add_ps(vl50, vl51);
    __m128 vh6 = _mm_add_ps(vh50, vh51);

    __m128 v7 = _mm_add_ps(vl6, vh6);

    return {v7[0]+v7[2], v7[1]+v7[3]};
  }

  // specialization for double for dumb compilers
  template<>
  inline std::complex<double> sum<std::complex<double>>(const Chunk<std::complex<double>>& v)
  {
    static_assert(Chunk<std::complex<double>>::size == 8);

    __m512d v02 = _mm512_load_pd(&v[0]);
    __m512d v82 = _mm512_load_pd(&v[4]);

    __m256d v030 = _mm512_extractf64x4_pd(v02, 0);
    __m256d v830 = _mm512_extractf64x4_pd(v82, 0);
    __m256d v031 = _mm512_extractf64x4_pd(v02, 1);
    __m256d v831 = _mm512_extractf64x4_pd(v82, 1);
    __m256d v04 = _mm256_add_pd(v030, v031);
    __m256d v84 = _mm256_add_pd(v830, v831);

    __m128d v050 = _mm256_extractf128_pd(v04, 0);
    __m128d v850 = _mm256_extractf128_pd(v84, 0);
    __m128d v051 = _mm256_extractf128_pd(v04, 1);
    __m128d v851 = _mm256_extractf128_pd(v84, 1);
    __m128d v06 = _mm_add_pd(v050, v051);
    __m128d v86 = _mm_add_pd(v850, v851);

    __m128d v9 = _mm_add_pd(v06, v86);

    return {v9[0], v9[1]};
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

  // specialization for double for dumb compilers
  template<>
  inline void bcast_sum<std::complex<float>>(Chunk<std::complex<float>>& v)
  {
    __m512 v0 = _mm512_add_ps(_mm512_load_ps(&v[0]), _mm512_load_ps(&v[8]));

    __m512 v3 = _mm512_permute_ps(v0, 2<<0 | 3<<2 | 0<<4 | 1<<6 );
    __m512 v4 = _mm512_add_ps(v0, v3);

    __m512 v5 = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), v4);
    __m512 v6 = _mm512_add_ps(v4, v5);

    __m512 v7 = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), v6);
    __m512 v8 = _mm512_add_ps(v6, v7);

    _mm512_store_ps(&v[0], v8);
    _mm512_store_ps(&v[8], v8);
  }

  // specialization for double for dumb compilers
  template<>
  inline void bcast_sum<std::complex<double>>(Chunk<std::complex<double>>& v)
  {
    __m512d v2 = _mm512_add_pd(_mm512_load_pd(&v[0]), _mm512_load_pd(&v[4]));

    __m512d v3 = _mm512_permutex_pd(v2, 2 | 3<<2 | 0<<4 | 1<<6);
    __m512d v4 = _mm512_add_pd(v2, v3);

    __m512d v5 = _mm512_permutexvar_pd(_mm512_set_epi64(3,2,1,0,7,6,5,4), v4);
    __m512d v6 = _mm512_add_pd(v4, v5);

    _mm512_store_pd(&v[0], v6);
    _mm512_store_pd(&v[4], v6);
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<float>(const Chunk<float>& src, short index, float value, Chunk<float>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long one = 1;
      __mmask16 mask = (one<<index)>>(16*i);
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
      unsigned long one = 1;
      __mmask8 mask = (one<<index)>>(8*i);
      __m512d xi = _mm512_load_pd(&src[8*i]);
      __m512d yi = _mm512_mask_broadcastsd_pd(xi, mask, _mm_set_pd(0,value));
      _mm512_store_pd(&result[8*i], yi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<std::complex<float>>(const Chunk<std::complex<float>>& src, short index, std::complex<float> value, Chunk<std::complex<float>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long three = 3;
      __mmask16 mask = (three<<(2*index))>>(16*i);
      __m512 xi = _mm512_load_ps(&src[8*i]);
      __m512 yi = _mm512_mask_broadcast_f32x2(xi, mask, _mm_set_ps(0,0,value.imag(),value.real()));
      _mm512_store_ps(&result[8*i], yi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void index_bcast<std::complex<double>>(const Chunk<std::complex<double>>& src, short index, std::complex<double> value, Chunk<std::complex<double>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long three = 3;
      __mmask8 mask = (three<<(2*index))>>(8*i);
      __m512d xi = _mm512_load_pd(&src[4*i]);
      __m512d yi = _mm512_mask_broadcast_f64x2(xi, mask, _mm_set_pd(value.imag(),value.real()));
      _mm512_store_pd(&result[4*i], yi);
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
  inline void masked_load_after<std::complex<float>>(const Chunk<std::complex<float>>& src, short index, Chunk<std::complex<float>>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask16 mask = (all<<(2*index))>>(16*i);
      __m512 vi = _mm512_maskz_load_ps(mask, &src[8*i]);
      _mm512_store_ps(&result[8*i], vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_load_after<std::complex<double>>(const Chunk<std::complex<double>>& src, short index, Chunk<std::complex<double>>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<(2*index))>>(8*i);
      __m512d vi = _mm512_maskz_load_pd(mask, &src[4*i]);
      _mm512_store_pd(&result[4*i], vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<std::complex<float>>(const Chunk<std::complex<float>>& src, short index, Chunk<std::complex<float>>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask16 mask = (all<<(2*index))>>(16*i);
      __m512 vi = _mm512_load_ps(&src[8*i]);
      _mm512_mask_store_ps(&result[8*i], mask, vi);
    }
  }

  // compilers seem not to generate masked SIMD commands
  template<>
  inline void masked_store_after<std::complex<double>>(const Chunk<std::complex<double>>& src, short index, Chunk<std::complex<double>>& result)
  {
    // of course, this code relies on the compiler optimization that eliminates all the redundant load/store operations
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      unsigned long all = -1; // set to 0xFF....
      __mmask8 mask = (all<<(2*index))>>(8*i);
      __m512d vi = _mm512_load_pd(&src[4*i]);
      _mm512_mask_store_pd(&result[4*i], mask, vi);
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

  // unaligned load
  template<>
  inline void unaligned_load<std::complex<float>>(const std::complex<float>* src, Chunk<std::complex<float>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_store_ps((float*)&result[8*i], _mm512_loadu_ps((const float*)&src[8*i]));
    }
  }

  // unaligned load
  template<>
  inline void unaligned_load<std::complex<double>>(const std::complex<double>* src, Chunk<std::complex<double>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_store_pd((double*)&result[4*i], _mm512_loadu_pd((const double*)&src[4*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<float>(const Chunk<float>& src, float* result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_storeu_ps(&result[16*i], _mm512_load_ps(&src[16*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<double>(const Chunk<double>& src, double* result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_storeu_pd(&result[8*i], _mm512_load_pd(&src[8*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<std::complex<float>>(const Chunk<std::complex<float>>& src, std::complex<float>* result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_storeu_ps((float*)&result[8*i], _mm512_load_ps((const float*)&src[8*i]));
    }
  }

  // unaligned store
  template<>
  inline void unaligned_store<std::complex<double>>(const Chunk<std::complex<double>>& src, std::complex<double>* result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_storeu_pd((double*)&result[4*i], _mm512_load_pd((const double*)&src[4*i]));
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

  // streaming store
  template<>
  inline void streaming_store<std::complex<float>>(const Chunk<std::complex<float>>& src, Chunk<std::complex<float>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_stream_ps((float*)&result[8*i], _mm512_load_ps((const float*)&src[8*i]));
    }
  }

  // streaming store
  template<>
  inline void streaming_store<std::complex<double>>(const Chunk<std::complex<double>>& src, Chunk<std::complex<double>>& result)
  {
    for(short i = 0; i < ALIGNMENT/64; i++)
    {
      _mm512_stream_pd((double*)&result[4*i], _mm512_load_pd((const double*)&src[4*i]));
    }
  }
}


#endif // PITTS_CHUNK_OPS_AVX512_HPP
