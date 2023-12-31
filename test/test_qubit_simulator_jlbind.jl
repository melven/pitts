#!/usr/bin/env julia

# Copyright (C) 2023 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PITTS::QubitSimulator Julia binding"""

import PittsJl
using Test

PittsJl.initialize(true, 12345)

@testset "Test PittsJl.QubitSimulator" begin

    @testset "simple example" begin
        using CxxWrap.StdLib: StdVector
        qsim = PittsJl.QubitSimulator()
        
        hadamardGate = PittsJl.Matrix2{Complex{Float64}}()
        hadamardGate[:,:] = Array{Complex{Float64}}([[1/sqrt(2.),1/sqrt(2.)] [1/sqrt(2.),-1/sqrt(2.)]])
        cnotGate = PittsJl.Matrix4{Complex{Float64}}()
        cnotGate[:,:] = Array{Complex{Float64}}([[1,0,0,0] [0,1,0,0] [0,0,0,1] [0,0,1,0]])

        for i = 1:10
            PittsJl.allocate_qubit(qsim, 7)
            PittsJl.allocate_qubit(qsim, 9)

            println(PittsJl.get_classical_value(qsim, 7, 1.e-10))

            PittsJl.apply_single_qubit_gate(qsim, 7, hadamardGate)
            PittsJl.apply_two_qubit_gate(qsim, 7, 9, cnotGate)

            println(PittsJl.measure_qubits(qsim, StdVector{UInt32}([7,9])))

            PittsJl.deallocate_qubit(qsim, 7)
            PittsJl.deallocate_qubit(qsim, 9)
        end

    end
end

PittsJl.finalize(true)
