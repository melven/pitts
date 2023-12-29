#!/usr/bin/env julia

# Copyright (C) 2023 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PITTS common functions Julia binding"""

import PittsJl
using Test


@testset "Test common" begin

    @test PittsJl.initialize(true, 12345) == nothing
    @test PittsJl.clearPerformanceStatistics() == nothing
    @test PittsJl.printPerformanceStatistics(true, true) == nothing
    @test PittsJl.finalize(true) == nothing

end