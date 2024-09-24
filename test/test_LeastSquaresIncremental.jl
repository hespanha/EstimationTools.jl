"""
Unit tests for LeastSquaresBatch.jl

2023 (C) Joao Hespanha
"""

using Random

using Test
using Printf
using BenchmarkTools

using LinearAlgebra
using Statistics

using EstimationTools

@testset "test leastSquares: test least squares with small regularization & reset!" begin
    Random.seed!(0)

    # identity X, square
    X = [1.0 0.0; 0.0 1.0]
    A = 2.0Matrix{Float64}(I, 2, 2)
    Y = A * X

    lambda = 1e-9

    (nX, nPoints) = size(X)
    nY = size(Y, 1)
    lsi = LSincremental(nX, nY, lambda)
    for i in 1:nPoints
        push!(lsi, X[:, i], Y[:, i])
    end
    @show hatA = leastSquares(lsi)
    @show norm(A - hatA)
    @test isapprox(hatA, A; rtol=1e-16)

    XX = hcat(X, 2 * X)
    YY = hcat(Y, 2 * Y)
    lsi = LSincremental(nX, nY, lambda)
    for i in 1:nPoints
        push!(lsi, X[:, i], Y[:, i])
    end
    @show hatAA = leastSquares(lsi)
    @show norm(A - hatAA)
    @test isapprox(hatAA, A; rtol=1e-16)

    X = rand(2, 2)
    A = rand(2, 2)
    Y = A * X
    lsi = LSincremental(nX, nY, lambda)
    for i in 1:nPoints
        push!(lsi, X[:, i], Y[:, i])
    end
    @show hatA = leastSquares(lsi)
    @show norm(A - hatA)
    #(hatA, report) = leastSquares(X, Y)
    @test isapprox(hatA, A; rtol=1e-5)

    XX = hcat(X, 2 * X)
    YY = hcat(Y, 2 * Y)
    lsi = LSincremental(nX, nY, lambda)
    for i in 1:nPoints
        push!(lsi, X[:, i], Y[:, i])
    end
    @show hatAA = leastSquares(lsi)
    @show norm(A - hatAA)
    @test isapprox(hatAA, A; rtol=1e-5)

    reset!(lsi, lambda)
    @test lsi.K == 0
    @test all(lsi.YX .== 0)
    @test lsi.R == 1 / lambda * I
end

@testset "test leastSquares: test in-place least squares with small regularization" begin
    Random.seed!(0)

    nX = 10
    nY = 5
    nPoints = 30

    # multiple matrices to estimate
    X = rand(Float64, nX, nPoints)
    As = [rand(nY, nX) for i in 1:2]
    Ys = [A * X for A in As]

    hatA = zeros(2 * nY, nX)
    lambda = 1e-9

    lsis = [LSincremental(nX, nY, lambda) for _ in As]
    for j in 1:2
        for i in 1:nPoints
            push!(lsis[j], X[:, i], Ys[j][:, i])
        end
        # store all estimates in one tall matrix
        hA = leastSquares!(@view(hatA[1+(j-1)*nY:j*nY, :]), lsis[j])
        @show norm(As[j] - hA)
    end

    for j in 1:2
        @show norm(As[j] - hatA[1+(j-1)*nY:j*nY, :])
        @test isapprox(As[j], hatA[1+(j-1)*nY:j*nY, :]; rtol=1e-5)
    end
end

@testset "test leastSquares: test least squares with large regularization" begin
    Random.seed!(0)

    # identity X, square
    X = [1.0 0.0; 0.0 1.0]
    A = 2.0Matrix{Float64}(I, 2, 2)
    Y = A * X
    lambda = 1e6

    (nX, nPoints) = size(X)
    nY = size(Y, 1)
    lsi = LSincremental(nX, nY, lambda)
    for i in 1:nPoints
        push!(lsi, X[:, i], Y[:, i])
    end
    @show hatA = leastSquares(lsi)
    @test norm(hatA) < 1e-5
end

## Test speed and allocations of LSdata

function baseline(
    lsd::LSdata{Float64,Int64},
    X::Matrix{Float64}, Y::Matrix{Float64},
    lambda::Float64,
)
    mul!(lsd.XX, X, X')
    mul!(lsd.YX, Y, X')
    mul!(lsd.YY, Y, Y')
    lsd.K = 0
    lsd.KK = size(X, 2)
    (hatA, report) = leastSquares(lsd; lambda, quiet=true)
end
function lsIncremental(
    lsi::LSincremental{Float64,Int64},
    X::Matrix{Float64}, Y::Matrix{Float64},
    lambda::Float64,
    hatA::Matrix{Float64},
)
    reset!(lsi, lambda)
    for k in axes(X, 2)
        x = @view X[:, k]
        y = @view Y[:, k]
        push!(lsi, x, y)
    end
    leastSquares!(hatA, lsi)
end

function test1()
    nX = 100
    nY = 50
    Kcache = 300
    lambda = 1e-6

    for nPoints in [200, 400, 1000]  # smaller and larger than cache
        @printf("\n# nPoints=%d, Kcache=%d\n", nPoints, Kcache)
        lsd = LSdata{Float64,Int64}(nX, nY, Kcache)
        lsi = LSincremental(nX, nY, lambda)
        hatA = Matrix{Float64}(undef, nY, nX)

        X = rand(Float64, nX, nPoints)
        Y = rand(Float64, nY, nPoints)

        #@time baseline(lsd, X, Y, lambda, hatA)
        #@time lsIncremental(lsi, X, Y, lambda, hatA)

        # Get base line times
        b0 = @benchmark baseline($lsd, $X, $Y, $lambda)
        #display(b0)
        # check time
        b1 = @benchmark lsIncremental($lsi, $X, $Y, $lambda, $hatA)
        #display(b1)
        @printf("   baseline time=%10.3f ms, LSincrement time=%10.3f ms, time slowdown up=%7.3f, LSincrement allocs=%d, memory=%d\n",
            1e-6 * Statistics.mean(b0.times), 1e-6 * Statistics.mean(b1.times),
            Statistics.mean(b1.times) / Statistics.mean(b0.times), b1.memory, b1.allocs)

        @test b1.memory == 0
        @test b1.allocs == 0
    end
end

@testset "test LSdata: speed and allocations" test1()
