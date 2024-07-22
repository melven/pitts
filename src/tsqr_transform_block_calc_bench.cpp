// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_common.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_tsqr_impl.hpp"
#include <charconv>
#include <iostream>
#include <stdexcept>


using Type = double;

namespace
{
  void transformBlock_calc2_BranchLess(int nChunks, const PITTS::MultiVector<Type>& Min1, const PITTS::MultiVector<Type>& Min2, PITTS::MultiVector<Type>& Mout, int nIter)
  {
    using namespace PITTS;

    // TODO: allow adding FMA and Mult
    const auto timer = PITTS::performance::createScopedTimer(
        {{"nChunks", "nIter"},{nChunks, nIter}}, // arguments
        //{{double(nChunks)*kernel_info::FMA<Chunk<Type>>() + nChunks+2.)*kernel_info::Mult<Chunk<Type>>()}, // flops - roughly estimated
        {{2*nIter*(nChunks + (nChunks+2)*0.5)*Chunk<Type>::size*kernel_info::FMA<Type>()}, // flops - roughly estimated
         {2*nIter*(nChunks+1.)*kernel_info::Load<Chunk<Type>>() + 2*nIter*(nChunks+2.)*kernel_info::Store<Chunk<Type>>()}} // data transfers
        );

    const MultiVector<Type>* pMin;
    for(int iter = 0; iter < nIter; iter++)
    {
      pMin = internal::randomGenerator() % 2 == 0 ? &Min1 : &Min2;
      PITTS::internal::HouseholderQR::transformBlock_calc<double,true>(nChunks, 2, &pMin->chunk(0,0), pMin->colStrideChunks(), &Mout.chunk(2,0), Mout.colStrideChunks(), 0, 0);
      pMin = internal::randomGenerator() % 2 == 0 ? &Min1 : &Min2;
      PITTS::internal::HouseholderQR::transformBlock_calc<double,true>(nChunks, 2, &pMin->chunk(0,0), pMin->colStrideChunks(), &Mout.chunk(2,0), Mout.colStrideChunks(), 0, 0);
    }
  }

  void transformBlock_calc2_WithBranches(int nChunks, const PITTS::MultiVector<Type>& Min1, const PITTS::MultiVector<Type>& Min2, PITTS::MultiVector<Type>& Mout, int nIter)
  {
    using namespace PITTS;

    // TODO: allow adding FMA and Mult
    const auto timer = PITTS::performance::createScopedTimer(
        {{"nChunks", "nIter"},{nChunks, nIter}}, // arguments
        //{{double(nChunks)*kernel_info::FMA<Chunk<Type>>() + nChunks+2.)*kernel_info::Mult<Chunk<Type>>()}, // flops - roughly estimated
        {{2*nIter*(nChunks + (nChunks+2)*0.5)*Chunk<Type>::size*kernel_info::FMA<Type>()}, // flops - roughly estimated
         {2*nIter*(nChunks+1.)*kernel_info::Load<Chunk<Type>>() + 2*nIter*(nChunks+2.)*kernel_info::Store<Chunk<Type>>()}} // data transfers
        );

    const MultiVector<Type>* pMin;
    for(int iter = 0; iter < nIter; iter++)
    {
      pMin = internal::randomGenerator() % 2 == 0 ? &Min1 : &Min2;
      PITTS::internal::HouseholderQR::transformBlock_calc<double,false>(nChunks, 2, &pMin->chunk(0,0), pMin->colStrideChunks(), &Mout.chunk(2,0), Mout.colStrideChunks(), 0, 0);
      pMin = internal::randomGenerator() % 2 == 0 ? &Min1 : &Min2;
      PITTS::internal::HouseholderQR::transformBlock_calc<double,false>(nChunks, 2, &pMin->chunk(0,0), pMin->colStrideChunks(), &Mout.chunk(2,0), Mout.colStrideChunks(), 0, 0);
    }
  }
}

int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 3 )
    throw std::invalid_argument("Requires 2 arguments (nChunks nIter)!");

  int nChunks = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], nChunks);
  std::from_chars(argv[2], argv[3], nIter);

  int n = PITTS::Chunk<Type>::size * (nChunks+3);

  PITTS::MultiVector<Type> Min(n, 2), Min2(n, 2), Mout(n, 2);
  randomize(Min);
  randomize(Min2);
  randomize(Mout);
  for(int i = 0; i < n; i++)
    Min2(i,0) = 0;

  transformBlock_calc2_BranchLess(nChunks, Min, Min2, Mout, 10);
  transformBlock_calc2_WithBranches(nChunks, Min, Min2, Mout, 10);


  PITTS::performance::clearStatistics();


  transformBlock_calc2_BranchLess(nChunks, Min, Min2, Mout, nIter);

  transformBlock_calc2_WithBranches(nChunks, Min, Min2, Mout, nIter);


  PITTS::finalize();

  return 0;
}

