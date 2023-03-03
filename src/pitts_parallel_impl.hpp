/*! @file pitts_parallel_impl.hpp
* @brief Helper functionality for parallelization (OpenMP, MPI)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_PARALLEL_IMPL_HPP
#define PITTS_PARALLEL_IMPL_HPP

// includes
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include "pitts_parallel.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper functionality for distribution of data and parallelization
    namespace parallel
    {
      // implement mpiGather for strings
      inline std::pair<std::string,std::vector<int>> mpiGather(const std::string_view& localData, int root, MPI_Comm comm)
      {
        const auto& [iProc,nProcs] = mpiProcInfo(comm);

        // gather sizes
        std::vector<int> sizes(nProcs,0);
        int localSize = localData.size();
        if( MPI_Gather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, root, comm) != MPI_SUCCESS )
          throw std::runtime_error("failure returned from MPI_Gather");

        // calculate offsets
        std::vector<int> offsets;
        offsets.reserve(nProcs+1);
        offsets.push_back(0);
        std::inclusive_scan(sizes.begin(), sizes.end(), std::back_inserter(offsets));

        // allocate buffer and gather data
        std::string globalData(offsets.back(), '\0');
        if( MPI_Gatherv(localData.data(), localData.size(), MPI_CHAR, globalData.data(), sizes.data(), offsets.data(), MPI_CHAR, root, comm) != MPI_SUCCESS )
          throw std::runtime_error("failure returned from MPI_Gatherv");

        // return the result
        return {std::move(globalData),std::move(offsets)};
      }


      // implement mpiCombineMaps
      template<typename Key, typename Value, class BinaryOperation>
      std::unordered_map<Key,Value> mpiCombineMaps(const std::unordered_map<Key,Value>& localMap, BinaryOperation op, int root, MPI_Comm comm)
      {
        // serialize data
        std::ostringstream oss;
        {
          cereal::BinaryOutputArchive ar(oss);
          ar( localMap );
        }

        const auto& [iProc,nProcs] = mpiProcInfo(comm);

        // communicate raw data: all processes -> root
        const auto& [rawData,offsets] = mpiGather(oss.str(), root, comm);

        // put result from all processes into a global map
        std::unordered_map<Key,Value> globalMap;
        if( iProc == root )
        {
          for(int i = 0; i < nProcs; i++)
          {
            std::unordered_map<Key,Value> otherMap;
            {
              std::istringstream iss(rawData.substr(offsets[i], offsets[i+1]-offsets[i]));
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


#endif // PITTS_PARALLEL_IMPL_HPP
