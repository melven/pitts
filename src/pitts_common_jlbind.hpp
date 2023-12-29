// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_common_jlbind.hpp
* @brief header for Julia binding for PITTS common helper functionality
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/


// include guard
#ifndef PITTS_COMMON_JLBIND_HPP
#define PITTS_COMMON_JLBIND_HPP

// includes
#include <jlcxx/jlcxx.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    //! create jlcxx-wrapper for PITTS::initialize and PITTS::finalize
    void define_common(jlcxx::Module& m);
  }
}

#endif // PITTS_COMMON_JLBIND_HPP
