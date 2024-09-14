"""
Main file to call all unit tests.

2023 (C) Joao Hespanha
"""

using EstimationTools
using Test

begin
    include("test_Estimators.jl")
    include("test_LeastSquares.jl")
end