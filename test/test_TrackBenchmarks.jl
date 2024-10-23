"""
Unit tests for TrackBenchmarks.jl

2024 (C) Joao Hespanha
"""

# TODO: test remove from IPsparse

using Test

using EstimationTools

@testset "saveBenchmark" begin

    d = Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9)
    d = Description(solveTime=0.1, solveTimeWithoutPrint=0.05, nIter=4)

    filename = "test/testTrackBenchmarks.csv"
    try
        run(`rm $filename`)
    catch err
        display(err)
    end

    @time df = saveBenchmark(
        filename,
        solver=Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9),
        problem=Description("Rock paper Scissors", nU=10, nEqU=1),
        time=Description(solveTime=0.1, solveTimeWithoutPrint=0.05, nIter=5),
        pruneBy=Minute(100))

    # not better
    @time df = saveBenchmark(
        filename,
        solver=Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9),
        problem=Description("Rock paper Scissors", nU=10, nEqU=1),
        time=Description(solveTime=0.1, solveTimeWithoutPrint=0.05, nIter=5),
        pruneBy=Minute(100))

    # better
    @time df = saveBenchmark(
        filename,
        solver=Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9),
        problem=Description("Rock paper Scissors", nU=10, nEqU=1),
        time=Description(solveTime=0.1, solveTimeWithoutPrint=0.05, nIter=4),
        pruneBy=Minute(100))

    # worse but different
    @time df = saveBenchmark(
        filename,
        solver=Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9),
        problem=Description("Rock paper Scissors", nU=10, nEqU=1),
        time=Description(solveTime=0.1, nIter=6.0),
        pruneBy=Minute(100))

    # better but different
    @time df = saveBenchmark(
        filename,
        solver=Description("Qminmax.jl", linearSolver=:LDL, equalityTolerance=1e-8, muFactorAggressive=0.9),
        problem=Description("Rock paper Scissors", nU=10, nEqU=1),
        time=Description(solveTime=0.1, nIter=5.0),
        pruneBy=Minute(100))

    @test size(df, 1) == 4

    @test df.timeValues == [
        [0.1, 0.05, 5.0],
        [0.1, 0.05, 4.0],
        [0.1, 6.0],
        [0.1, 5.0]]

end