module ConvergenceLogging

export TimeSeriesLogger, plotLogger!, plotLogger

using Statistics
using ElasticArrays
using Plots

"""
Structure to store convergence data for plotting.
"""
struct TimeSeriesLogger{T,D}
    time::ElasticVector{T}
    data::ElasticMatrix{D}
    maxPoints::Int
    xlabel::String
    ylabel::String
    legend::Vector{String}
    ylimits::Vector{Float64}
    function TimeSeriesLogger{T,D}(N::Int;
        maxPoints::Int=200,
        xlabel::String="time",
        ylabel::String="data",
        legend::Vector{String}=string.(1:N),
        ylimits::Vector{Float64}=[-NaN64, NaN64],
        #ylimits::Vector{Float64}=[-1.0, 1.0],
    ) where {T,D}
        time::ElasticVector{T} = ElasticVector{T}(undef, 0)
        data::ElasticMatrix{D} = ElasticMatrix{D}(undef, N, 0)
        return new{T,D}(time, data, maxPoints,
            xlabel, ylabel, legend, ylimits)
    end
end

function Base.append!(
    clog::TimeSeriesLogger{T,D},
    t::T,
    d::AbstractVector{D},
) where {T,D}
    append!(clog.time, t)
    append!(clog.data, d)
end

"""Plot array of loggers, at each subplot."""
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    clog::Vector{TimeSeriesLogger{T,D}};
) where {T,D}
    for i in eachindex(clog)
        plotLogger!(plt, i, clog[i])
    end
end
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    subplot::AbstractVector{Int},
    clog::Vector{TimeSeriesLogger{T,D}};
) where {T,D}
    for i in eachindex(subplot, clog)
        plotLogger!(plt, subplot[i], clog[i])
    end
end

"""Plot logger at a subplot"""
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    subplot::Int,
    clog::TimeSeriesLogger{T,D};
) where {T,D}

    plt[subplot].series_list = [] # new one to erase old points
    color_series = palette(:tab20)
    nPoints = length(clog.time)
    (t::Vector{T}, me::Matrix{Float64}, mn::Matrix{Float64}, mx::Matrix{Float64}) = subSample(clog)
    if nPoints < 50
        for d in axes(me, 2)
            Plots.plot!(plt[subplot], t, me[:, d], linecolor=color_series[d],
                ylimits=clog.ylimits,
                xlabel=clog.xlabel, ylabel=clog.ylabel,
                labels=clog.legend[d], #legend=:topleft,
                markershape=:circle, markerstrokewidth=0,
                grid=true)
        end
    else
        for d in axes(me, 2)
            Plots.plot!(plt[subplot], t, me[:, d], linecolor=color_series[d],
                ylimits=clog.ylimits,
                xlabel=clog.xlabel, ylabel=clog.ylabel, labels=clog.legend[d],
                grid=true)
        end
    end
    for d in axes(me, 2)
        k = .!(isnan.(mn[:, d]) .|| isnan.(mx[:, d]) .|| isinf.(mn[:, d]) .|| isinf.(mx[:, d]))
        if any(k)
            Plots.plot!(plt[subplot], t[k], mn[k, d], fillrange=mx[k, d], linecolor=color_series[d],
                c=1, fillalpha=0.1, linealpha=0,
                label="")
        end
    end
    return plt
end
function plotLogger(
    clog::TimeSeriesLogger{T,D}
) where {T,D}
    plt = Plots.plot()
    return plotLogger!(Plots.plot(), 1, clog)
end

function subSample(
    clog::TimeSeriesLogger{T,D},
) where {T,D}
    time = clog.time
    data = clog.data
    (nD, nPoints) = size(data)
    maxPoints = min(clog.maxPoints, nPoints)
    t = Vector{T}(undef, maxPoints)
    me = Matrix{Float64}(undef, maxPoints, nD)
    mn = Matrix{Float64}(undef, maxPoints, nD)
    mx = Matrix{Float64}(undef, maxPoints, nD)
    stp::Int = ceil(nPoints / maxPoints)
    i = 1
    for j in eachindex(t)
        t[j] = time[min(i, nPoints)]
        for k in 1:nD
            iData = @view data[k, i:min(i + stp - 1, nPoints)]
            noNaN = Iterators.filter(!isnan, iData)
            me[j, k] = mean(iData)
            mn[j, k] = minimum(noNaN; init=+Inf64)
            mx[j, k] = maximum(noNaN; init=-Inf64)
        end
        i += stp
    end
    return (t, me, mn, mx)
end

end