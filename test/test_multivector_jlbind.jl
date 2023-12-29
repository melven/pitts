#!/usr/bin/env julia

# Copyright (C) 2023 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PITTS::MultiVector Julia binding"""

import PittsJl
using Test


for T = (Float32, Float64, Complex{Float32}, Complex{Float64})
    @testset "Test PittsJl.MultiVector{$(repr(T))}" begin

        @testset "construct_empty" begin
            mv = PittsJl.MultiVector{T}()
            @test size(mv) == (0,0)
        end

        @testset "construct_with_size" begin
            mv = PittsJl.MultiVector{T}(10,2)
            @test size(mv) == (10,2)
        end

        @testset "resize" begin
            mv = PittsJl.MultiVector{T}(5,7)
            resize!(mv, 37, 3, true, false)
            @test size(mv) == (37,3)
        end

        @testset "randomize" begin
            mv = PittsJl.MultiVector{T}(13,2)
            PittsJl.randomize!(mv)
            @test size(mv) == (13,2)
            @test mv[1,1] != mv[2,2]
        end

        @testset "copy" begin
            mv1 = PittsJl.MultiVector{T}(13,2)
            PittsJl.randomize!(mv1)
            mv2 = PittsJl.MultiVector{T}()
            PittsJl.copy(mv1, mv2)
            @test mv1 == mv2
        end

        @testset "centroids" begin
            using CxxWrap: StdVector, CxxLongLong

            X = PittsJl.MultiVector{T}(20,5)
            Y = PittsJl.MultiVector{T}(20,2)
            PittsJl.randomize!(X)
            PittsJl.randomize!(Y)
          
            idx = StdVector{CxxLongLong}([1, 0, 0, 1, 0])
            w = StdVector{T}([44., 55., 66., 77., 88.])
          
            PittsJl.centroids(X, idx, w, Y);
          
            for i = 1:20
                @test 55*X[i,2] + 66*X[i,3] + 88*X[i,5] ≈ Y[i,1]
                @test 44*X[i,1] + 77*X[i,4] ≈ Y[i,2]
            end
        end

        @testset "cdist2" begin
            import LinearAlgebra

            X = PittsJl.MultiVector{T}(20,5)
            Y = PittsJl.MultiVector{T}(20,2)

            PittsJl.randomize!(X)
            PittsJl.randomize!(Y)
          
            dist = PittsJl.Tensor2{T}()
            PittsJl.cdist2(X, Y, dist);

            dist_ref = zeros(T,5,2)
            for i = 1:5
                for j = 1:2
                    dist_ref[i,j] = LinearAlgebra.norm(X[:,i] - Y[:,j])^2
                end
            end

            @test dist ≈ dist_ref
        end

        @testset "block_TSQR" begin
            import LinearAlgebra

            X = PittsJl.MultiVector{T}(10,3)
            R = PittsJl.Tensor2{T}()

            PittsJl.randomize!(X)
            PittsJl.block_TSQR(X, R, 0, false, 0)

            X_svd = LinearAlgebra.svd(X)
            R_svd = LinearAlgebra.svd(R)

            @test R_svd.S ≈ X_svd.S
            for i = 1:3
                @test abs(LinearAlgebra.dot(R_svd.Vt[i,:], X_svd.Vt[i,:])) ≈ 1
            end
        end

        @testset "transform" begin
            
            X = PittsJl.MultiVector{T}(10,3)
            M = PittsJl.Tensor2{T}(3,3)
            Y = PittsJl.MultiVector{T}(10,3)

            PittsJl.randomize!(X)
            PittsJl.randomize!(M)
            PittsJl.randomize!(Y)

            PittsJl.transform(X, M, Y, 0, 0)

            @test Y ≈ X * M
        end

        @testset "transpose" begin
            
            X = PittsJl.MultiVector{T}(10,3)
            Y = PittsJl.MultiVector{T}()

            PittsJl.randomize!(X)

            PittsJl.transpose(X, Y, 0, 0, false)

            @test Y ≈ transpose(X)
        end
    end
end
