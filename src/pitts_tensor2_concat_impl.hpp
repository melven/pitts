// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_concat_impl.hpp
* @brief Concatenize 2 tensors to one larger tensor (combine blocks of matrices)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-25
*
**/

// include guard
#ifndef PITTS_TENSOR2_CONCAT_IMPL_HPP
#define PITTS_TENSOR2_CONCAT_IMPL_HPP

// includes
#include <stdexcept>
#include "pitts_tensor2_concat.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement tensor2 concatLeftRight
  template<typename T>
  void concatLeftRight(const std::optional<ConstTensor2View<T>>& A, const std::optional<ConstTensor2View<T>>& B, Tensor2View<T> C)
  {
    const auto r1 = C.r1();
    const auto r2 = C.r2();
    if( A && A->r1() != r1 )
      throw std::invalid_argument("concatLeftRight: mismatching dimension (A.r1() != C.r1())");
    if( B && B->r1() != r1 )
      throw std::invalid_argument("concatLeftRight: mismatching dimension (B.r1() != C.r1())");
    const auto rA2 = A ? A->r2() : B ? r2 - B->r2() : r2;
    if( rA2 > r2 || rA2 < 0 )
      throw std::invalid_argument("concatLeftRight: mismatching dimension (A.r2() + B.r2() != C.r2())");
    const auto rB2 = r2 - rA2;
    if( B && B->r2() != rB2 )
      throw std::invalid_argument("concatLeftRight: mismatching dimension (A.r2() + B.r2() != C.r2())");

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*r2*kernel_info::NoOp<T>()},    // flops
         {r1*r2*kernel_info::Store<T>() + (((bool)A)*r1*rA2 + ((bool)B)*r1*rB2)*kernel_info::Load<T>()}}  // data
        );
    
#pragma omp parallel for collapse(2) schedule(static) if(r1*rA2 > 100)
    for(long long j = 0; j < rA2; j++)
      for(long long i = 0; i < r1; i++)
        C(i,j) = A ? (*A)(i,j) : T(0);

#pragma omp parallel for collapse(2) schedule(static) if(r1*rB2 > 100)
    for(long long j = 0; j < rB2; j++)
      for(long long i = 0; i < r1; i++)
        C(i,rA2+j) = B ? (*B)(i,j) : T(0);
  }


  // implement tensor2 concatTopBottom
  template<typename T>
  void concatTopBottom(const std::optional<ConstTensor2View<T>>& A, const std::optional<ConstTensor2View<T>>& B, Tensor2View<T> C)
  {
    const auto r1 = C.r1();
    const auto r2 = C.r2();
    if( A && A->r2() != r2 )
      throw std::invalid_argument("concatTopBottom: mismatching dimension (A.r2() != C.r2())");
    if( B && B->r2() != r2 )
      throw std::invalid_argument("concatTopBottom: mismatching dimension (B.r2() != C.r2())");
    const auto rA1 = A ? A->r1() : B ? r1 - B->r1() : r1;
    if( rA1 > r1 || rA1 < 0 )
      throw std::invalid_argument("concatLeftRight: mismatching dimension (A.r2() + B.r2() != C.r2())");
    const auto rB1 = r1 - rA1;
    if( B && B->r1() != rB1 )
      throw std::invalid_argument("concatTopBottom: mismatching dimension (A.r1() + B.r1() != C.r1())");

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*r2*kernel_info::NoOp<T>()},    // flops
         {r1*r2*kernel_info::Store<T>() + (((bool)A)*rB1*r2 + ((bool)B)*rB1*r2)*kernel_info::Load<T>()}}  // data
        );
    
#pragma omp parallel for collapse(2) schedule(static) if(rA1*r2 > 100)
    for(long long j = 0; j < r2; j++)
      for(long long i = 0; i < rA1; i++)
        C(i,j) = A ? (*A)(i,j) : T(0);

#pragma omp parallel for collapse(2) schedule(static) if(rB1*r2 > 100)
    for(long long j = 0; j < r2; j++)
      for(long long i = 0; i < rB1; i++)
        C(rA1+i,j) = B ? (*B)(i,j) : T(0);
  }

}


#endif // PITTS_TENSOR2_CONCAT_IMPL_HPP
