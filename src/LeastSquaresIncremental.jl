module LeastSquaresIncremental

export LSincremental, leastSquares!

using LinearAlgebra

import EstimationTools: leastSquares, reset! # to add methods

##################################
#### Batch least squares - storage
##################################

"""
   Structure used to store data to trains *incrementally* a linear model of the form
        y_k = A x_k + noise    k in 1:K
    using least-squares.

    The "batch" solution is of the form

        hat A = Z_K * R_K

    where 

        Z_K = sum_{k=1]^K y_k x_k'
        R_K = (regularizer + sum_{k=1}^K y_k x_k')^{-1}
        regularizer = lambda*I

    However, the R_K can be updated incrementally using the Sherman-Morrison formula:

        R_{k+1} = R_k -  (R_k x_{k+1}) (R_k x_{k+1})' / (1+ x_{k+1}'R_kx_{k+1})

# Fields:
- `R::Matrix{FloatLS}`: matrix R_K 
- `K::IntLS`: number of data points 
- `YX::Matrix{FloatLS}`: matrix Z_K

# Parameters for constructor:
- `nX::IntLS`: size of the vectors x_k
- `nY::IntLS`: size of the vectors y_k
- `lambda::FloatLS`: regularization parameter

# Attention:

1) The regularization parameter is fixed and remains the same regardless of the number of points,
   which means that its effect vanishes as the number of points increases.

   This may be desireable.

2) The field `K:IntLS` is not really used.
"""
mutable struct LSincremental{FloatLS,IntLS}
    R::Matrix{FloatLS}
    YX::Matrix{FloatLS}
    K::IntLS
    buffer_Rx::Vector{FloatLS}
    LSincremental(nX::IntLS, nY::IntLS, lambda::FloatLS) where {FloatLS,IntLS} =
        new{FloatLS,IntLS}(
            Matrix{FloatLS}((1 / lambda) * I, nX, nX),
            zeros(FloatLS, nY, nX),
            IntLS(0),
            Vector{FloatLS}(undef, nX)
        )
end

"""
    reset!(lsi::LSincremental{FloatLS,IntLS},lambda::FloatLS)

Clear all data from `LSincremental`

# Parameters:
- `lsi::LSincremental{FloatLS,IntLS}`: structure with data
- `lambda::FloatLS`: regularization parameter
"""
function reset!(
    lsi::LSincremental{FloatLS,IntLS},
    lambda::FloatLS,
) where {FloatLS,IntLS}
    fill!(lsi.R, 0)
    lambda = 1 / lambda
    for i in 1:size(lsi.R, 1)
        lsi.R[i, i] = lambda
    end
    lsi.K = 0
end

"""
    push!(lsi::LSincremental{FloatLS,IntLS},x::Matrix{FloatLS},y::Matrix{FloatLS})

Adds data vectors x and y to `LSincremental`.

# Parameters:
- `lsi::LSincremental{FloatLS,IntLS}`: structure with data
- `x::AbstractVector{FloatLS}`: input vector
- `y::AbstractVector{FloatLS}`: output vector
"""
function Base.push!(
    lsi::LSincremental{FloatLS,IntLS},
    x::AbstractVector{FloatLS},
    y::AbstractVector{FloatLS},
) where {FloatLS,IntLS}
    lsi.K += 1
    # YX = YX + y*x'
    mul!(lsi.YX, y, x', one(FloatLS), one(FloatLS))

    # R = R - (R*x)*(R*x)' / (1+x*R*X)
    mul!(lsi.buffer_Rx, lsi.R, x)
    #@show lsi.buffer_Rx
    alpha = -one(FloatLS) / (one(FloatLS) + dot(x, lsi.buffer_Rx))
    #@show alpha
    mul!(lsi.R, lsi.buffer_Rx, lsi.buffer_Rx', alpha, one(FloatLS))
    #@show lsi.R
    return nothing
end

"""
    A=leastSquares!(lsi)

Computes least-squares estimate.

# Parameters:
- `lsi::LSincremental{FloatLS,IntLS}`: structure with data

# Returns:
- `hatA::AbstractMatrix{FloatLS}`: least-squares estimate
"""
function leastSquares(
    lsi::LSincremental{FloatLS,IntLS},
) where {FloatLS,IntLS}
    return lsi.YX * lsi.R
end

"""
    leastSquares!(A,lsi)

Computes least-squares estimate in place

# Parameters:
- `hatA::AbstractMatrix{FloatLS}`: least-squares estimate, returned in place
- `lsi::LSincremental{FloatLS,IntLS}`: structure with data

# Attention: 

    If passing a submatrix as `hatA` a @view is required, as in
        leastSquares!( (@view hatA[1:nY,1:nY]), lsi)
    Otherwise the submatrix is not re-written.
"""
function leastSquares!(
    hatA::AbstractMatrix{FloatLS},
    lsi::LSincremental{FloatLS,IntLS},
) where {FloatLS,IntLS}
    mul!(hatA, lsi.YX, lsi.R)
    return hatA
end
end