"""
Main file to call all unit tests.

2023 (C) Joao Hespanha
"""

using EstimationTools
using Test

begin
    include("test_Estimators.jl")
    include("test_LeastSquaresBatch.jl")
    include("test_LeastSquaresIncremental.jl")
end