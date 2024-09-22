module ConvergenceLogging

export TimeSeriesLogger, plotLogger!, plotLogger

using Statistics
using ElasticArrays
using Plots

"""
Structure used to store convergence data for plotting.

    logger=TimeSeriesLogger{T,D}(N; maxPoints,xlabel,ylabel,legend,ylimits)

# Fields:
+ `time::ElasticVector{T}`: time vector
+ `data::ElasticMatrix{D}`: matrix with data to plot, with one time instant per column and one time
        series per row
+ `maxPoints::Int`: maximum number of points to plot. When the number of time instants is larger
        than this value, multiple time instants are "aggregates" and represented by their mean, min,
        and max values.
+ `xlabel::String="time"`: label for the x axis
+ `ylabel::String="data"`: label for the y axis
+ `legend::Vector{String}=string.(1:N)`: legend for the different time series
+ `ylimits::Vector{Float64}=[-NaN64, NaN64]`: limits for the y axis

# Constructor parameters:
+ `T::Type`: type of the time variable
+ `D::Type`: type of the data variables
+ `N::Int`: number of time series
+ `maxPoints::Int`: maximum number of points to plot. When the number of time instants is larger
        than this value, multiple time instants are "aggregates" and represented by their mean, min,
        and max values.
+ `xlabel::String="time"`: label for the x axis
+ `ylabel::String="data"`: label for the y axis
+ `legend::Vector{String}=string.(1:N)`: legend for the different time series
+ `ylimits::Vector{Float64}=[-NaN64, NaN64]`: limits for the y axis
"""
struct TimeSeriesLogger{T,D}
    time::ElasticVector{T}
    data::ElasticMatrix{D}
    maxPoints::Int
    xlabel::String
    ylabel::String
    legend::Vector{String}
    ylimits::Vector{Float64}
    function TimeSeriesLogger{T,D}(
        N::Int;
        maxPoints::Int=200,
        xlabel::String="time",
        ylabel::String="data",
        legend::Vector{String}=string.(1:N),
        ylimits::Vector{Float64}=[-NaN64, NaN64],
    ) where {T,D}
        time::ElasticVector{T} = ElasticVector{T}(undef, 0)
        data::ElasticMatrix{D} = ElasticMatrix{D}(undef, N, 0)
        return new{T,D}(time, data, maxPoints,
            xlabel, ylabel, legend, ylimits)
    end
end

"""
Add one data point to the logger.

# Parameters:
+ `logger::TimeSeriesLogger{T,D}`: logger
+ `t::T`: time 
+ `d::AbstractVector{D}`: values
"""
function Base.append!(
    logger::TimeSeriesLogger{T,D},
    t::T,
    d::AbstractVector{D},
) where {T,D}
    append!(logger.time, t)
    append!(logger.data, d)
    return nothing
end

"""
Plot array of loggers, each in one subplot of a given plot.

# Parameters:
+ `plt::Union{Plots.Plot,Plots.Subplot}`: Plot, typically with multiple subplots
+ 'subplot::AbstractVector{Int}': Optional vector with the indices of the subplots to use. When
        missing, the first subplots are used.
+ `logger::Vector{TimeSeriesLogger{T,D}}`: Vector of loggers to plot
"""
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    logger::Vector{TimeSeriesLogger{T,D}};
) where {T,D}
    for i in eachindex(logger)
        plotLogger!(plt, i, logger[i])
    end
    return nothing
end
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    subplot::AbstractVector{Int},
    logger::Vector{TimeSeriesLogger{T,D}};
) where {T,D}
    for i in eachindex(subplot, logger)
        plotLogger!(plt, subplot[i], logger[i])
    end
    return nothing
end

"""
Plot one logger in one subplot of a given plot.

# Parameters:
+ `plt::Union{Plots.Plot,Plots.Subplot}`: Plot, typically with multiple subplots
+ 'subplot::Int': Index of the subplots to use.
+ `logger::TimeSeriesLogger{T,D}`: logger to plot
"""
function plotLogger!(
    plt::Union{Plots.Plot,Plots.Subplot},
    subplot::Int,
    logger::TimeSeriesLogger{T,D};
    #palette::Symbol=:viridis, # progressive
    colors::Symbol=:glasbey_category10_n256, # categorical
) where {T,D}
    plt[subplot].series_list = [] # new one to erase old points
    #color_series = palette(:tab20, size(me, 2))
    (nD, nPoints) = size(logger.data)
    color_series = palette(colors, nD)
    (t::Vector{T}, me::Matrix{Float64}, mn::Matrix{Float64}, mx::Matrix{Float64}) = subSample(logger)
    if nPoints < 50
        for d in axes(me, 2)
            Plots.plot!(plt[subplot], t, me[:, d], linecolor=color_series[d],
                ylimits=logger.ylimits,
                xlabel=logger.xlabel, ylabel=logger.ylabel, labels=logger.legend[d],
                markershape=:circle, markerstrokewidth=0, markercolor=color_series[d],
                grid=true)
        end
    else
        for d in axes(me, 2)
            Plots.plot!(plt[subplot], t, me[:, d], linecolor=color_series[d],
                ylimits=logger.ylimits,
                xlabel=logger.xlabel, ylabel=logger.ylabel, labels=logger.legend[d],
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
"""
Plot one logger in a new plot.

# Parameters:
+ `logger::TimeSeriesLogger{T,D}`: logger to plot
"""
function plotLogger(
    logger::TimeSeriesLogger{T,D}
) where {T,D}
    plt = Plots.plot()
    return plotLogger!(Plots.plot(), 1, logger)
end

"""Subsample logger times to the desired number of points."""
function subSample(
    logger::TimeSeriesLogger{T,D},
) where {T,D}
    time = logger.time
    data = logger.data
    (nD, nPoints) = size(data)
    maxPoints = min(logger.maxPoints, nPoints)
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