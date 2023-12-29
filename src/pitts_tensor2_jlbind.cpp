// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_jlbind.cpp
* @brief Julia binding for PITTS::Tensor2
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
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"

// add inheritance hierarchy infos
namespace jlcxx
{
  template<> struct SuperType<PITTS::Tensor2View<float>> { typedef PITTS::ConstTensor2View<float> type; };
  template<> struct SuperType<PITTS::Tensor2View<double>> { typedef PITTS::ConstTensor2View<double> type; };
  template<> struct SuperType<PITTS::Tensor2View<std::complex<float>>> { typedef PITTS::ConstTensor2View<std::complex<float>> type; };
  template<> struct SuperType<PITTS::Tensor2View<std::complex<double>>> { typedef PITTS::ConstTensor2View<std::complex<double>> type; };

  template<> struct SuperType<PITTS::Tensor2<float>> { typedef PITTS::Tensor2View<float> type; };
  template<> struct SuperType<PITTS::Tensor2<double>> { typedef PITTS::Tensor2View<double> type; };
  template<> struct SuperType<PITTS::Tensor2<std::complex<float>>> { typedef PITTS::Tensor2View<std::complex<float>> type; };
  template<> struct SuperType<PITTS::Tensor2<std::complex<double>>> { typedef PITTS::Tensor2View<std::complex<double>> type; };
}

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    // create jlcxx-wrapper for PITTS::Tensor2
    void define_Tensor2(jlcxx::Module& m)
    {
      auto const_tensor2_view = m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("ConstTensor2View", jlcxx::julia_type("AbstractMatrix"))
        .apply<PITTS::ConstTensor2View<float>, PITTS::ConstTensor2View<double>, PITTS::ConstTensor2View<std::complex<float>>, PITTS::ConstTensor2View<std::complex<double>>>([](auto wrapped)
        {
          using ConstTensor2View_type = decltype(wrapped)::type;
          using Type = ConstTensor2View_type::Type;

          // array interface
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("size", [](const ConstTensor2View_type& t2){return std::tuple<jlcxx::cxxint_t,jlcxx::cxxint_t>(t2.r1(), t2.r2());});
          wrapped.template method("getindex", [](const ConstTensor2View_type& t2, jlcxx::cxxint_t i, jlcxx::cxxint_t j) -> Type {return t2(i-1,j-1);});
          wrapped.module().unset_override_module();
        });

      auto tensor2_view = m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("Tensor2View", const_tensor2_view.dt())
        .apply<PITTS::Tensor2View<float>, PITTS::Tensor2View<double>, PITTS::Tensor2View<std::complex<float>>, PITTS::Tensor2View<std::complex<double>>>([](auto wrapped)
        {
          using Tensor2View_type = decltype(wrapped)::type;
          using Type = Tensor2View_type::Type;

          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("setindex!", [](Tensor2View_type& t2, Type val, jlcxx::cxxint_t i, jlcxx::cxxint_t j){t2(i-1,j-1) = val;});
          wrapped.module().unset_override_module();

          wrapped.template method("randomize!", static_cast<void (*)(Tensor2View_type&)>(&randomize));
        });

      m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("Tensor2", tensor2_view.dt())
        .apply<PITTS::Tensor2<float>, PITTS::Tensor2<double>, PITTS::Tensor2<std::complex<float>>, PITTS::Tensor2<std::complex<double>>>([](auto wrapped)
        {
          using Tensor2_type = decltype(wrapped)::type;
          using Type = Tensor2_type::Type;
          wrapped.template constructor<long long, long long>();

          // array interface
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("resize!", &Tensor2_type::resize);
          wrapped.module().unset_override_module();

          wrapped.template method("copy", static_cast<void (*)(const Tensor2_type&, Tensor2_type&)>(&copy));
        });

/*
      m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("Tensor2", jlcxx::julia_type("AbstractMatrix"))
        .apply<PITTS::Tensor2<float>, PITTS::Tensor2<double>, PITTS::Tensor2<std::complex<float>>, PITTS::Tensor2<std::complex<double>>>([](auto wrapped)
        {
          using Tensor2_type = decltype(wrapped)::type;
          using Type = Tensor2_type::Type;
          wrapped.template constructor<long long, long long>();

          // array interface
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("resize!", &Tensor2_type::resize);
          wrapped.template method("size", [](const Tensor2_type& t2){return std::tuple<jlcxx::cxxint_t,jlcxx::cxxint_t>(t2.r1(), t2.r2());});
          wrapped.template method("getindex", [](const Tensor2_type& t2, jlcxx::cxxint_t i, jlcxx::cxxint_t j) -> Type {return t2(i-1,j-1);});
          wrapped.template method("setindex!", [](Tensor2_type& t2, Type val, jlcxx::cxxint_t i, jlcxx::cxxint_t j){t2(i-1,j-1) = val;});
          wrapped.module().unset_override_module();

          wrapped.template method("copy", static_cast<void (*)(const Tensor2_type&, Tensor2_type&)>(&copy));
          wrapped.template method("randomize!", static_cast<void (*)(Tensor2_type&)>(&randomize));
        });
*/
    }
  }
}
