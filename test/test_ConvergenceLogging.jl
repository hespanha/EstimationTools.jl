using Revise

using Plots
using Test

using EstimationTools

Plots.closeall()

begin
    function testSpeed()
        N = 1_000
        t = 1:N

        @time a = 0.5 .+ 10 * randn(9, N)
        @time sa = cumsum(a, dims=2)

        #VSCodeServer.PLOT_PANE_ENABLED[] = true
        #VSCodeServer.PLOT_PANE_ENABLED[] = false
        display(VSCodeServer.PLOT_PANE_ENABLED[])

        println(:png)
        @time plt2 = plot(t, sa', fmt=:png)
        @time display(plt2)
        sleep(5.0)

        println(:svg)
        @time plt1 = plot(t, sa', fmt=:svg)
        @time display(plt1)
        sleep(5.0)

        return nothing
    end

    testSpeed()
end

@testset "ConvergenceLogging: small number of points" begin
    @time clog = TimeSeriesLogger{Int,Float64}(5)
    @test length(clog.time) == 0
    @test size(clog.data) == (5, 0)

    @time append!(clog, 1, Float64[1, 3, 4, 5, 6])
    @test length(clog.time) == 1
    @test size(clog.data) == (5, 1)

    @time append!(clog, 3, 2 .* Float64[1, 3, 4, 5, 6])
    @test length(clog.time) == 2
    @test size(clog.data) == (5, 2)

    l = @layout [a b]
    plts = plot(layout=l)
    @time plotLogger!(plts, 1, clog)
    @time plotLogger!(plts, 2, clog, colors=:temperaturemap)
    for k in 1:4
        append!(clog, 4 + k, (-1) .^ k .* Float64[1, 3, 4, 5, 6])
        @time plotLogger!(plts, 1, clog)
        @time plotLogger!(plts, 2, clog, colors=:temperaturemap)
        display(plts)
        sleep(0.5)
    end
    @test length(clog.time) == 6
    @test size(clog.data) == (5, 6)
end

@testset "ConvergenceLogging: many points with bands"

begin
    @time clog = TimeSeriesLogger{Int,Float64}(3)
    @test length(clog.time) == 0
    @test size(clog.data) == (3, 0)

    N = 1000
    for t in 1:1000
        append!(clog, t, rand(3) .+ [2.0, 3.5, 5])
    end

    l = @layout [a; b]
    plts = plot(layout=l)
    @time plotLogger!(plts, 1, clog)
    @time plotLogger!(plts, 2, clog, colors=:temperaturemap)

end

