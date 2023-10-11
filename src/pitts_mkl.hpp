/*! @file pitts_mkl.hpp
* @brief Helper functions and quirks (for non-Intel CPUs) for using Intel MKL
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-09-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MKL_HPP
#define PITTS_MKL_HPP

// includes
#ifdef PITTS_DIRECT_MKL_GEMM
#include <mkl_cblas.h>
#endif

//! workaround for using Intel MKL on non-Intel CPUs
extern "C" int mkl_serv_intel_cpu_true() {return true;}


#endif // PITTS_MKL_HPP
