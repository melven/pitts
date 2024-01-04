#!/usr/bin/env julia

# Copyright (C) 2023 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PITTS::Tensor2 Julia binding"""

import PittsJl
using Test

PittsJl.initialize(true, 12345)

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

        @testset "ArrayInterface" begin
            A = PittsJl.Tensor2{T}(5,1)
            A[1,1] = T(2)
            @test A[1,1] == 2
            A[5,1] = T(42)
            @test A[5,1] == 42

            A .= T(1)

            B = PittsJl.Tensor2{T}(1,1)
            B .= T(2)

            @test A * B ≈ [2, 2, 2, 2, 2]
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

        @testset "split_normalize_qb" begin
            using LinearAlgebra: I

            M = PittsJl.Tensor2{T}(13, 5)
            PittsJl.randomize!(M)

            AB = PittsJl.split_normalize_qb(M, true, T(0), 999, false)
            A = copy(AB[1])
            B = copy(AB[2])

            @test A * B ≈ M
            @test adjoint(A) * A ≈ I

            AB = PittsJl.split_normalize_qb(M, false, T(0), 999, false)
            A = copy(AB[1])
            B = copy(AB[2])

            @test A * B ≈ M
            @test B * adjoint(B) ≈ I
        end

        @testset "split_normalize_svd" begin
            using LinearAlgebra: I

            M = PittsJl.Tensor2{T}(13, 5)
            PittsJl.randomize!(M)

            AB = PittsJl.split_normalize_svd(M, true, T(0), 999, false, false, T(0))
            A = copy(AB[1])
            B = copy(AB[2])

            #@test isapprox(A * B, M, atol=0.001, rtol=0.001)
            @test A * B ≈ M
            @test adjoint(A) * A ≈ I

            AB = PittsJl.split_normalize_svd(M, false, T(0), 999, false, false, T(0))
            A = copy(AB[1])
            B = copy(AB[2])

            #@test isapprox(A * B, M, atol=0.001, rtol=0.001)
            @test A * B ≈ M
            @test B * adjoint(B) ≈ I
        end
    end
end

PittsJl.finalize(true)
