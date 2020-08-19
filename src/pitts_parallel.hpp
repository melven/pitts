/*! @file pitts_parallel.hpp
* @brief Helper functionality for parallelization (OpenMP, MPI)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_PARALLEL_HPP
#define PITTS_PARALLEL_HPP

// includes
#include <mpi.h>
#include <unordered_map>
#include <numeric>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper functionality for distribution of data and parallelization
    namespace parallel
    {
      //! helper function to combine std::unordered_map from multiple MPI processes
      //!
      //! @tparam Key               std::unordered_map key type, must be serializable using cereal
      //! @tparam Value             std::unordered_map value type, must be serializable using cereal
      //! @tparam BinaryOperation   function signature to combine two values, defaults to + operator
      //!
      //! @param localMap           std::unordered_map on each process
      //! @param op                 (optional) operator to combine values of keys that occur on multiple processes, only relevant on process 0
      //! @param comm               (optional) MPI communicator, must be identical on all MPI processes
      //!
      template<typename Key, typename Value, class BinaryOperation = std::function<Value(Value,Value)>>
      std::unordered_map<Key,Value> combineMaps(const std::unordered_map<Key,Value>& localMap, BinaryOperation op = std::plus<Value>(), MPI_Comm comm = MPI_COMM_WORLD)
      {
        // serialize data
        std::ostringstream oss;
        {
          cereal::BinaryOutputArchive ar(oss);
          ar( localMap );
        }

        // get number of processes
        int nProcs = 1, iProc = 0;
        if( MPI_Comm_size(comm, &nProcs) != 0 )
          throw std::runtime_error("MPI error");
        if( MPI_Comm_rank(comm, &iProc) != 0 )
          throw std::runtime_error("MPI error");

        // communicate raw data: all processes -> root
        std::vector<int> sizes(nProcs, 0);
        int localSize = oss.str().size();
        if( MPI_Gather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, comm) != 0 )
          throw std::runtime_error("MPI error");
        std::vector<int> offsets;
        std::exclusive_scan(sizes.begin(), sizes.end(), std::back_inserter(offsets), 0);
        int totalSize = offsets.back() + sizes.back();
        std::string raw(totalSize, '\0');
        if( MPI_Gatherv(oss.str().data(), oss.str().size(), MPI_CHAR, raw.data(), sizes.data(), offsets.data(), MPI_CHAR, 0, comm) != 0 )
          throw std::runtime_error("MPI error");

        // put result from all processes into a global map
        std::unordered_map<Key,Value> globalMap;
        if( iProc == 0 )
        {
          for(int i = 0; i < nProcs; i++)
          {
            std::unordered_map<Key,Value> otherMap;
            {
              std::istringstream iss(raw.substr(offsets[i], sizes[i]));
              cereal::BinaryInputArchive ar(iss);
              ar( otherMap );
            }
            for(const auto& [key,val]: otherMap)
            {
              // try to insert the new key, merge key using op when it's already there
              auto [iter,didInsert] = globalMap.insert({key,val});
              if( !didInsert )
                iter->second = op(iter->second, val);
            }
          }
        }
        return globalMap;
      }

    }
  }
}


#endif // PITTS_PARALLEL_HPP
