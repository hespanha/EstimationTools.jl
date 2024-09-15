"""
Unit tests for LeastSquares.jl

2023 (C) Joao Hespanha
"""

using Test
using Printf
using BenchmarkTools

using LinearAlgebra
using Statistics

using EstimationTools

@testset "test leastSquares: test least squares without regularization" begin
    # identity X, square
    X = [1.0 0.0; 0.0 1.0]
    A = 2.0Matrix{Float64}(I, 2, 2)
    Y = A * X
    (hatA, report) = leastSquares(X, Y, lambda=0.0)
    @test hatA ≈ A

    XX = hcat(X, 2 * X)
    YY = hcat(Y, 2 * Y)
    (hatAA, report) = leastSquares(XX, YY)
    @test hatAA ≈ A

    X = rand(2, 2)
    A = rand(2, 2)
    Y = A * X
    (hatA, report) = leastSquares(X, Y)
    @test hatA ≈ A

    XX = hcat(X, 2 * X)
    YY = hcat(Y, 2 * Y)
    (hatAA, report) = leastSquares(XX, YY)
    @test hatAA ≈ A
end

@testset "test leastSquares: test least squares with regularization" begin
    # identity X, square
    X = [1.0 0.0; 0.0 1.0]
    A = 2.0Matrix{Float64}(I, 2, 2)
    Y = A * X
    (hatA, report) = leastSquares(X, Y, lambda=1e6)
    @test norm(hatA) < 1e-5
end

@testset "test leastSquares: using LSdata" begin
    # identity X, square
    X = [1.0 0.0; 0.0 1.0]
    A = 2.0Matrix{Float64}(I, 2, 2)
    Y = A * X

    nX = size(X, 1)
    nY = size(Y, 1)
    Kcache = 100
    lsd = LSdata{Float64,Int64}(nX, nY, Kcache)
    for i in 1:size(X, 2)
        push!(lsd, X[:, i], Y[:, i])
    end
    (hatA, report) = leastSquares(lsd; verbose=true, lambda=0.0)
    @test hatA ≈ A
end


## Test speed and allocations of LSdata

function baseline(lsd::LSdata{Float64,Int64}, X::Matrix{Float64}, Y::Matrix{Float64})
    mul!(lsd.XX, X, X')
    mul!(lsd.YX, Y, X')
    mul!(lsd.YY, Y, Y')
end
function lsData(lsd::LSdata{Float64,Int64}, X::Matrix{Float64}, Y::Matrix{Float64})
    reset!(lsd)
    for k in axes(X, 2)
        x = @view X[:, k]
        y = @view Y[:, k]
        push!(lsd, x, y)
    end
    compress!(lsd)
end

function test1()
    nX = 100
    nY = 50
    Kcache = 300

    for K in [200, 400, 1000]  # smaller and larger than cache
        @printf("K=%d, Kcache=%d\n", K, Kcache)
        lsd = LSdata{Float64,Int64}(nX, nY, Kcache)
        @test size(lsd.X) == (nX, Kcache)
        @test size(lsd.Y) == (nY, Kcache)
        @test lsd.K == 0
        @test size(lsd.XX) == (nX, nX)
        @test size(lsd.YX) == (nY, nX)
        @test size(lsd.YY) == (nY, nY)
        @test lsd.KK == 0

        X = rand(Float64, nX, K)
        Y = rand(Float64, nY, K)

        # check correctness
        lsData(lsd, X, Y)
        if K <= size(lsd.X, 2)
            @test lsd.X[:, 1:K] == X
            @test lsd.Y[:, 1:K] == Y
        end
        @test norm(lsd.XX - X * X') < 1e-8
        @test norm(lsd.YX - Y * X') < 1e-8
        @test norm(lsd.YY - Y * Y') < 1e-8

        # Get base line times
        b0 = @benchmark baseline($lsd, $X, $Y)
        # check time
        b1 = @benchmark lsData($lsd, $X, $Y)
        @printf("   time slowdown up=%g, allocs=%d memory=%d\n",
            Statistics.mean(b1.times) / Statistics.mean(b0.times), b1.memory, b1.allocs)

        @test b1.memory == 0
        @test b1.allocs == 0
    end
end

@testset "test LSdata: speed and allocations" test1()
