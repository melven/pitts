// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_jlbind.cpp
* @brief Julia binding for PITTS::MultiVector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <jlcxx/array.hpp>
#include <string>
#include <stdexcept>
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_jlbind.hpp"
#include "pitts_scope_info.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    // create jlcxx-wrapper for PITTS::TensorTrain
    void define_MultiVector(jlcxx::Module& m)
    {
      // add code for std::vector of std::complex
      jlcxx::stl::apply_stl<std::complex<float>>(m);
      jlcxx::stl::apply_stl<std::complex<double>>(m);

      m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("MultiVector", jlcxx::julia_type("AbstractMatrix"))
        .apply<PITTS::MultiVector<float>, PITTS::MultiVector<double>, PITTS::MultiVector<std::complex<float>>, PITTS::MultiVector<std::complex<double>>>([](auto wrapped)
        {
          using MultiVector_type = decltype(wrapped)::type;
          using Type = MultiVector_type::Type;
          wrapped.template constructor<long long, long long>();

          // array interface
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("resize!", &MultiVector_type::resize);
          wrapped.template method("size", [](const MultiVector_type& mv){return std::tuple<jlcxx::cxxint_t,jlcxx::cxxint_t>(mv.rows(), mv.cols());});
          wrapped.template method("getindex", [](const MultiVector_type& mv, jlcxx::cxxint_t i, jlcxx::cxxint_t j) -> Type {return mv(i-1,j-1);});
          wrapped.template method("setindex!", [](MultiVector_type& mv, Type val, jlcxx::cxxint_t i, jlcxx::cxxint_t j){mv(i-1,j-1) = val;});
          wrapped.module().unset_override_module();

          wrapped.template method("copy", static_cast<void (*)(const MultiVector_type&, MultiVector_type&)>(&copy));
          wrapped.template method("randomize!", static_cast<void (*)(MultiVector_type&)>(&randomize));
          wrapped.template method("centroids", &centroids<Type>);
          wrapped.template method("cdist2", &cdist2<Type>);
          wrapped.template method("block_TSQR", &block_TSQR<Type>);
          wrapped.template method("transform", [](const MultiVector_type& X, const ConstTensor2View<Type>& M, MultiVector_type& Y, long long rows, long long cols)
          {
            transform(X, M, Y, {rows, cols});
          });
          wrapped.template method("transpose", [](const MultiVector_type& X, MultiVector_type& Y, long long rows, long long cols, bool reverse)
          {
            transpose(X, Y, {rows, cols}, reverse);
          });
        });
    }
  }
}
