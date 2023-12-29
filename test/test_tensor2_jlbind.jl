#!/usr/bin/env julia

# Copyright (C) 2023 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PITTS::Tensor2 Julia binding"""

import PittsJl
using Test


for T = (Float32, Float64, Complex{Float32}, Complex{Float64})
    @testset "Test PittsJl.Tensor2{$(repr(T))}" begin

        @testset "construct_empty" begin
            mv = PittsJl.Tensor2{T}()
            @test size(mv) == (0,0)
        end

        @testset "construct_with_size" begin
            mv = PittsJl.Tensor2{T}(10,2)
            @test size(mv) == (10,2)
        end

        @testset "resize" begin
            mv = PittsJl.Tensor2{T}(5,7)
            resize!(mv, 37, 3, true)
            @test size(mv) == (37,3)
        end

        @testset "randomize" begin
            mv = PittsJl.Tensor2{T}(13,2)
            PittsJl.randomize!(mv)
            @test size(mv) == (13,2)
            @test mv[1,1] != mv[2,2]
        end

        @testset "copy" begin
            mv1 = PittsJl.Tensor2{T}(13,2)
            PittsJl.randomize!(mv1)
            mv2 = PittsJl.Tensor2{T}()
            PittsJl.copy(mv1, mv2)
            @test mv1 == mv2
        end
    end
end
