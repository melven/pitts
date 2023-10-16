// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_parallel_impl.hpp"
#include "pitts_timer.hpp"
#include "pitts_performance.hpp"

using namespace PITTS;


using Map1 = std::unordered_map<std::string, PITTS::internal::TimingStatistics>;
using Map2 = std::unordered_map<std::string, PITTS::internal::PerformanceStatistics>;
using AddFunction = decltype(std::plus());

template Map1 PITTS::internal::parallel::mpiCombineMaps<std::string, PITTS::internal::TimingStatistics, AddFunction>(const Map1&, AddFunction, int, MPI_Comm);
template Map2 PITTS::internal::parallel::mpiCombineMaps<std::string, PITTS::internal::PerformanceStatistics, AddFunction>(const Map2&, AddFunction, int, MPI_Comm);
