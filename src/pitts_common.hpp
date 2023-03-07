/*! @file pitts_common.hpp
* @brief Useful helper functionality, e.g. MPI/OpenMP initialization
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-25
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_COMMON_HPP
#define PITTS_COMMON_HPP

// includes
#include "pitts_random.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Call MPI_Init if needed and print some general informations
  //!
  //! @param argc       pointer to main argc argument for MPI_Init
  //! @param argv       pointer to main argv argument for MPI_Init
  //! @param verbose    set to false to omit any console output (info on #threads and #processes)
  //! @param randomSeed seed for the random number generator used in PITTS
  //!
  void initialize(int* argc, char** argv[], bool verbose = true, std::uint_fast64_t randomSeed = internal::generateRandomSeed());


  //! Call MPI_Finalize if needed and print some statistics
  //!
  //! @param verbose set to false to omit any console output (for timing statistics)
  //!
  void finalize(bool verbose = true);
}


#ifndef PITTS_DEVELOP_BUILD
#include "pitts_common_impl.hpp"
#endif

#endif // PITTS_COMMON_HPP
