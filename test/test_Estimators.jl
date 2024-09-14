"""
Unit tests for Estimators.jl

2023 (C) Joao Hespanha
"""

using Test

import Statistics
using EstimationTools

begin
    estimator = Estimator()
    for x in range(0.0, 10.0, 11)
        add!(estimator, x)
    end

    @show EstimationTools.mean(estimator)
    @show EstimationTools.median(estimator)
    @show EstimationTools.minmax(estimator)
    @show EstimationTools.std(estimator)

    @show EstimationTools.meanCI(estimator, 0.95)
    @show EstimationTools.rangeCI(estimator, 0.95)
    @show EstimationTools.rangeCI(estimator, 1.0 - 2 / 10) # take 2 out of range of 10

    @test EstimationTools.mean(estimator) == 5.0
    @test EstimationTools.median(estimator) == 5.0
    @test EstimationTools.minmax(estimator) == (0.0, 10.0)
    @test EstimationTools.std(estimator) == Statistics.std(range(0.0, 10.0, 11), corrected=false)

    @test all(isapprox.(EstimationTools.meanCI(estimator, 0.95),
        EstimationTools.mean(estimator) .+
        2.228 * EstimationTools.std(estimator) / sqrt(11) * [-1, 1]; atol=1e-3))

    @test all(isapprox.(EstimationTools.rangeCI(estimator, 1.0 - 2 / 10), (1.0, 9.0); atol=1e-8))
end
