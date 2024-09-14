module LeastSquares

export LSreport, LSreports
export LSdata, compress!, reset!
export leastSquares, SSD

using LinearAlgebra
using DataStructures # needed for SortedDict

using Printf

using TimerOutputs

## Debug
debugLevel = 6

#######################################
#### Least squares estimation - reports
#######################################

"""
    LSreport

Report with status from least-squares estimation, indexed by report parameters.

# Example

    report::LSreport = Dict(
            "msqe" => msqe,
            "msqy" => msqy,
            "# equations" => nEquations,
            "# unknowns" => nUnknowns,
            "lambda" => lambda,
            "regularizer" => [minimum(diag(regularizer)) maximum(diag(regularizer))],
            "time" => dt,
            "alloc" => db,)
"""
const LSreport = SortedDict{String,Union{Int64,Float64,Matrix{Float64}}}    # sorted by key order, requires DataStructures.jl
#const LSreport = Dict{String,Float64}          # regular Dictionaries
#const LSreports = Dict{String,LSreport}       # regular Dictionaries
#const LSreport = OrderedDict{String,Float64}    # sorted by insertion order, requires DataStructures.jl
#const LSreports = OrderedDict{String,LSreport} # sorted by insertion order, requires DataStructures.jl

"""
    LSreports

Collection of reports with status from multiple least-squares estimations, indexed by estimation
description.

Example

    reports::LSreports = Dict(
        "dynamics" => Dict(
            "msqe" => rmseDynamicsFul,
            "msqy" => rmsyDynamics,),
        "outputs" => Dict(
            "msqe" => rmseOutput,
            "msqy" => rmsyOutput,),
    )
"""
const LSreports = SortedDict{String,LSreport} # sorted by key order, requires DataStructures.jl

function Base.display(report::LSreport; indent::Integer=0, brief::Bool=false)
    indentSpaces = repeat(" ", indent)
    if haskey(report, "msqe") && haskey(report, "msqy")
        if brief
            @printf(" rmse = %.6f (%.3f %%)\n",
                sqrt(report["msqe"]), 100 * sqrt(report["msqe"] / report["msqy"]))
            return nothing
        else
            @printf("\n%s%-20s = %.6f (%.3f %%)\n", indentSpaces, "rmse",
                sqrt(report["msqe"]), 100 * sqrt(report["msqe"] / report["msqy"]))
        end
    else
        println()
    end
    for (k, v) in report
        if isa(v, Number)
            if v < 0.001
                @printf("%s%-20s = %.3e\n", indentSpaces, k, v)
            else
                @printf("%s%-20s = %.6f\n", indentSpaces, k, v)
            end
        else
            @printf("%s%-20s = %s\n", indentSpaces, k, string(v))
        end
    end
    return nothing
end

function Base.display(reports::LSreports; indent::Integer=0, brief::Bool=false)
    indentSpaces = repeat(" ", indent)
    for (k, v) in reports
        @printf("%s%-80s", indentSpaces, k)
        display(v; indent=indent + 3, brief)
    end
end

#######################################
#### Least squares estimation - storage
#######################################

"""
   Structure used to store data to trains a linear model of the form
        y_k = A x_k + noise    k in 1:K
    using least-squares.

# Fields:
- `X::Matrix{FloatLS}`: matrix with one x_k per column
- `Y::Matrix{FloatLS}`: matrix with one y_k per column
- `K::IntLS`: number of valid vectors present in X and Y
- `XX::Matrix{FloatLS}`: matrix with sum_k x_k x_k'
- `YX::Matrix{FloatLS}`: matrix with sum_k y_k x_k'
- `YY::Matrix{FloatLS}`: matrix with sum_k y_k y_k'
- `KK::IntLS`: number of summations in XX, YX, and YY

   Terms that have already been included into XX,YX,YY,KK do not appear in X,Y,K

# Parameters for constructor:
- `nX::IntLS`: size of the vectors x_k
- `nY::IntLS`: size of the vectors y_k
- `Kcache::IntLS`: number of columns reserved for the matrices X, Y
"""
mutable struct LSdata{FloatLS,IntLS}
    X::Matrix{FloatLS}
    Y::Matrix{FloatLS}
    K::IntLS
    XX::Matrix{FloatLS}
    YX::Matrix{FloatLS}
    YY::Matrix{FloatLS}
    KK::IntLS
    LSdata{FloatLS,IntLS}(nX::IntLS, nY::IntLS, Kcache::IntLS) where {FloatLS,IntLS} =
        new{FloatLS,IntLS}(
            zeros(FloatLS, nX, Kcache),
            zeros(FloatLS, nY, Kcache),
            IntLS(0),
            zeros(FloatLS, nX, nX),
            zeros(FloatLS, nY, nX),
            zeros(FloatLS, nY, nY),
            IntLS(0),
        )
end

"""
    compress(lsd::LSdata{FloatLS,IntLS})

Moves data in `LSdata` stored in X,Y,K into XX,YX,YY,KK
"""
function compress!(
    lsd::LSdata{FloatLS,IntLS},
) where {FloatLS,IntLS}
    if lsd.K == size(lsd.X, 2)
        mul!(lsd.XX, lsd.X, lsd.X', 1.0, 1.0)
        mul!(lsd.YX, lsd.Y, lsd.X', 1.0, 1.0)
        mul!(lsd.YY, lsd.Y, lsd.Y', 1.0, 1.0)
    else
        X = @view lsd.X[:, 1:lsd.K]
        Y = @view lsd.Y[:, 1:lsd.K]
        mul!(lsd.XX, X, X', 1.0, 1.0)
        mul!(lsd.YX, Y, X', 1.0, 1.0)
        mul!(lsd.YY, Y, Y', 1.0, 1.0)
    end
    lsd.KK += lsd.K
    lsd.K = 0
    return lsd
end
"""
    reset!(lsd::LSdata{FloatLS,IntLS})

Clear all data from `LSdata`
"""
function reset!(
    lsd::LSdata{FloatLS,IntLS},
) where {FloatLS,IntLS}
    lsd.K = 0
    fill!(lsd.XX, 0)
    fill!(lsd.YX, 0)
    fill!(lsd.YY, 0)
    lsd.KK = 0
end

"""
    push!(lsd::LSdata{FloatLS,IntLS},x::Matrix{FloatLS},y::Matrix{FloatLS})

Adds data vectors x and y to `LSdata`, compressing it if X,Y are full
"""
function Base.push!(
    lsd::LSdata{FloatLS,IntLS},
    x::Matrix{FloatLS},
    y::Matrix{FloatLS},
) where {FloatLS,IntLS}
    if lsd.K >= size(lsd.X, 2)
        compress!(lsd)
    end
    lsd.K += 1
    lsd.X[:, lsd.K] .= x
    lsd.Y[:, lsd.K] .= y
    return lsd
end
function Base.push!(
    lsd::LSdata{FloatLS,IntLS},
    x::Vector{FloatLS},
    y::Vector{FloatLS},
) where {FloatLS,IntLS}
    if lsd.K >= size(lsd.X, 2)
        compress!(lsd)
    end
    lsd.K += 1
    lsd.X[:, lsd.K] .= x
    lsd.Y[:, lsd.K] .= y
    return lsd
end
function Base.push!(
    lsd::LSdata{FloatLS,IntLS},
    x::SubArray{FloatLS,1,Matrix{FloatLS},Tuple{Base.Slice{Base.OneTo{IntLS}},IntLS},true},
    y::SubArray{FloatLS,1,Matrix{FloatLS},Tuple{Base.Slice{Base.OneTo{IntLS}},IntLS},true},
) where {FloatLS,IntLS}
    if lsd.K >= size(lsd.X, 2)
        compress!(lsd)
    end
    lsd.K += 1
    lsd.X[:, lsd.K] = x
    lsd.Y[:, lsd.K] = y
    return lsd
end

########################################
#### Least squares estimation - solution
########################################

"""
    A=leastSquares(X::Matrix{FloatLS},Y::Matrix{FloatLS};lambda::FloatLS=convert(FloatLS,0.0))

Estimates the matrix A in the model
    y_k = A x_k + noise    k in 1:K
where A is a solution to the following least-squares problem
    minimize_D   1/K sum_k ||y_k - A x_k||^2 + lambda ||A||_fro^2

# Parameters:
- `X::Matrix{FloatLS}(nX,nSamples)`: matrix with independent variables
            [x_1 x_2 ... x_K]
- `Y::Matrix{FloatLS}(nY,nSamples)`: dependent variable
            [y_1 y_2 ... y_K]
- `lambda::FloatLS=convert(FloatLS,0.0)`: regularization parameter
- `uniformRegularization::Bool=false`: if `true`, regularization is based in scaled identity; otherwise a general diagonal matrix
- `regularization::Bool=true`: if `true` uses regularization by adding a penalty term
- `method4LSQ::Symbol=:rdiv`: method uses to compute least-squares solution, among the options
        + `:rdiv`: uses `\` to invert (X * X' + regularization matrix)
        + `:pinv`: uses `\` to invert (X * X' + regularization matrix)
        + `:svd`: uses a SVD decomposition to invert X*X' (ignoring the very small singular values)
        + `:eig`: uses an eigenvalue decomposition to invert X*X' (ignoring the very small eigenvalues)
- `computeErrorVariance::Bool=false`: when `true` include error variances for all entries of the model in the report

# Returns:
- `A::Matrix{FloatLS}`: estimated matrix
- `report::LSreport`
"""
function leastSquares(
    X::Matrix{FloatLS},
    Y::Matrix{FloatLS};
    timerOutput=nothing,
    verbose=false,
    useSSD::Bool=false,
    kwargs...
)::Tuple{Matrix{FloatLS},LSreport} where {FloatLS}

    ownTimer::Bool = isnothing(timerOutput)
    if isnothing(timerOutput)
        timerOutput = TimerOutput()
    end

    t0 = Base.time_ns()

    (nX, K) = size(X)
    @assert any(isnan.(Y)) == false "leastSquares: Y cannot have NaN"
    @assert any(isnan.(X)) == false "leastSquares: X cannot have NaN"
    @assert K > 0 "leastSquares: data set cannot be empty"

    @timeit timerOutput "XXK" XXK = (X * X') / K
    @timeit timerOutput "YXK" YXK = (Y * X') / K
    @timeit timerOutput "YYK" YYK = Y * Y' / K

    rc = leastSquares(XXK, YXK, YYK, K; timerOutput, verbose, kwargs...)

    dt = 1e-9 * (Base.time_ns() - t0)
    if verbose || (dt > 1.0 && ownTimer)
        #TimerOutputs.complement!(timerOutput) # add missing times
        show(timerOutput, sortby=:firstexec, compact=true)
        println()
    end

    return rc
end

# TODO: not really used, but ony works when nX==nY
function SSD(
    X::Matrix{FloatLS},
    Y::Matrix{FloatLS};
    timerOutput::TimerOutput=TimerOutput()
) where {FloatLS}
    iteration = 1
    tol = 0
    while true
        (nX, K) = size(X)
        @printf("SSD[%d]: nX=%4d, K=%d\n", iteration, nX, K)
        @timeit timerOutput "svd" (U, s, V) = svd(hcat(X', Y'), full=false)
        # "full decomposition"
        #     hcat(X', Y') _{K , 2 nX}
        #     U_{K,K}    s_{K,2nX}  V_{2nX,2nX}
        # "economic decomposition for K > 2nX with 
        #     U_{K,2nX}  s_{2nX}  V_{2nX,2nX} 
        @show size(U)
        @show size(s)
        @show size(V)
        @show minimum(s)
        @show maximum(s)
        if tol == 0
            tol = 1e-3 * maximum(s)
        end
        k = s .< tol
        @show sum(k)
        if !any(k)
            @printf("SSD: kernel has dimension %4d, only trivial (zero) invariant subspace\n", sum(k))
            X = Matrix{FloatLS}(undef, nX, 0)
            Y = Matrix{FloatLS}(undef, nX, 0)
            return (X, Y)
        elseif nX <= sum(k)
            @printf("SSD: kernel has dimension %4d >= %4d, exiting with nX=%4d, K=%d\n", sum(k), nX, nX, K)
            return (X, Y)
        end
        ZX = V[1:nX, k]'
        X = ZX * X
        Y = ZX * Y
        iteration += 1
    end
end

"""
    YYK = Y * Y' / K
    XXK = (X * X') / K
    YXK = (Y * X') / K
"""
function leastSquares(
    XXK::Matrix{FloatLS},
    YXK::Matrix{FloatLS},
    YYK::Matrix{FloatLS},
    K::Integer;
    lambda::FloatLS=convert(FloatLS, 0.0),
    uniformRegularization::Bool=false,
    #regularization::Bool=false,
    #method4LSQ::Symbol=:eig,    # in [:rdiv, :pinv, :svd, :eig]
    regularization::Bool=true,
    method4LSQ::Symbol=:rdiv,    # in [:rdiv, :pinv, :svd, :eig]
    computeErrorVariance::Bool=false,
    quiet=false,
    verbose=false,
    timerOutput::TimerOutput,
) where {FloatLS<:Real}

    t0 = Base.time_ns()
    b0 = Base.gc_bytes()

    (nY, nX) = size(YXK)
    nEquations = nY * K
    nUnknowns = nY * nX

    if regularization
        if uniformRegularization
            ## scaled identity
            #    normXXK = norm(diag(XXK))
            #    normXXK = minimum(abs.(diag(XXK))) # not good if zero in diagonal
            normXXK = mean(abs.(diag(XXK)))
            if normXXK > 0
                lambdaScaled = lambda * normXXK # scale lambda
            else
                lambdaScaled = lambda
            end
            regularizer = lambdaScaled * Matrix{FloatLS}(I, nX, nX)
        else
            ## diagonal
            regularizer = diagm(lambda * max.(diag(XXK), 1e-3))
        end
    else
        regularizer = zeros(FloatLS, size(XXK))
    end

    ## Estimate with regularization
    if method4LSQ == :pinv
        @timeit timerOutput "pinv" XX1 = pinv(Symmetric(XXK + regularizer))
        @timeit timerOutput "A" A = YXK * XX1
        note::String = "pinv"
    elseif method4LSQ == :rdiv
        @timeit timerOutput "rdiv" A = YXK / Symmetric(XXK + regularizer)
        note = "rdiv"
    elseif method4LSQ == :svd
        @timeit timerOutput "svd" (U, s, V) = svd(Symmetric(XXK))
        k = s .> lambda / 2000
        A = YXK * U[:, k] * diagm(1.0 ./ s[k]) * (V'[k, :])
        note = @sprintf("svd:%d/%d", sum(k), length(s))
    elseif method4LSQ == :eig
        @timeit timerOutput "eigen" F = eigen(Symmetric(XXK))
        k = F.values .> lambda / 2000
        A = YXK * F.vectors[:, k] * diagm(1.0 ./ F.values[k]) * F.vectors[:, k]'
        note = @sprintf("eig:%d/%d", sum(k), length(F.values))
    else
        error(" unknown method4LSQ = $method4LSQ")
    end

    ## Estimate noise
    @timeit timerOutput "noise estimate" begin
        # hat Sigma_noise = 1/K (Y-AX)(Y-AX)' 
        #                 = YYK + A XXK A' - YXK A' - A YXK'
        noiseVariance = -YXK * A'
        noiseVariance += noiseVariance' + YYK + A * XXK * A'
    end

    if computeErrorVariance
        if lambda > 0
            @warn(@sprintf("leastSquares: error variance estimate only accurate for lambda=0 (not %g)", lambda))
        end
        ## error variance estimate
        @timeit timerOutput "error variance estimate" begin
            @timeit timerOutput "pinv" XX1 = pinv(XXK)
            errorVariance = reshape(diag(noiseVariance), :, 1) * reshape(diag(XX1), 1, :)
        end
    end

    ## Compute msqe
    @timeit timerOutput "msqe" begin
        # error2 =trace (AX-Y)'(AX-Y)=trace (AX-Y)(AX-Y)'=trace (AXX'A'+YY'-AXY'-YX'A)
        msqy = tr(YYK)
        # abs should not be needed, but included because of numerical errors
        msqe = abs(tr(noiseVariance))
    end

    dt = 1e-9 * (Base.time_ns() - t0)
    db = Base.gc_bytes() - b0
    if verbose || (!quiet &&
                   (debugLevel > 8 ||
                    (debugLevel > 5 &&
                     (nEquations < 2 * nUnknowns || sqrt(msqe / msqy) > 0.05))) # too few equations or large errors
    )
        if msqy > 0
            @printf("      lsq(%s): nY=%4d,nX=%4d,K=%6d, lmb=%7.1e=>(%7.1e,%7.1e), eq=%9d, unk=%7d [%5.1f%%]->rmse=%5.2f%% (%.3fs)\n", #", %.0fMb)\n",
                note,
                nY, nX, K,
                lambda, minimum(diag(regularizer)), maximum(diag(regularizer)),
                nEquations, nUnknowns, 100 * nUnknowns / nEquations,
                100 * sqrt(msqe / msqy), dt)#, 1e-6 * db)
        else
            @printf("      lsq(%s): nY=%4d,nX=%4d,K=%6d, lmb=%7.1e=>(%7.1e,%7.1e), eq=%9d, unk=%7d [%5.1f%%]->rmse=%5.2e (%.3fs)\n", #", %.0fMb)\n",
                note,
                nY, nX, K,
                lambda, minimum(diag(regularizer)), maximum(diag(regularizer)),
                nEquations, nUnknowns, 100 * nUnknowns / nEquations,
                100 * sqrt(msqe), dt)#, 1e-6 * db)
        end
    end
    report::LSreport = Dict(
        "msqe" => msqe,
        "msqy" => msqy,
        "# equations" => nEquations,
        "# unknowns" => nUnknowns,
        "lambda" => lambda,
        "regularizer" => [minimum(diag(regularizer)) maximum(diag(regularizer))],
        "time" => dt,
        "alloc" => db,
    )
    if computeErrorVariance
        report["noise covariance"] = noiseVariance
        report["estimates"] = A
        report["error variance"] = errorVariance
    end
    return (A::Matrix{FloatLS}, report)
end



end