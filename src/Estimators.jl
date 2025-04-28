module Estimators

export Estimator, add!, mean, median, std, meanCI, rangeCI
export actualVSforecast45deg

using Statistics
using Distributions
using Printf

"""
Utility functions to help
+ evaluate optimization algorithms
+ validate models
"""

###################################
### Scalar mean-variance estimation
###################################

#using PythonCall

import Statistics
using LinearAlgebra

"""
Estimator for scalar variables of:
+ mean
+ std. dev.
+ confidence intervals for the mean
+ confidence intervals for the range

# Available methods:
+ `add!(estimator::Estimator, value::Int) = add!(estimator, Float64(value))`
+ `EstimationTools.mean(estimator::Estimator)`
+ `(min,max)=Base.minmax(estimator::Estimator)`
+ `EstimationTools.std(estimator::Estimator, me::Float64)`
+ `EstimationTools.median(estimator::Estimator)`
+ `EstimationTools.meanCI(estimator::Estimator, percent::AbstractFloat)`
+ `(lower,upper)=EstimationTools.range(estimator::Estimator, percent::AbstractFloat)`
+ `Base.display(estimator::Estimator)`
"""
mutable struct Estimator
    count::Int64
    sum::Float64
    sum2::Float64
    min::Float64
    max::Float64
    values::Vector{Float64}
    Estimator(;
        count::Int64=0,
        sum::Float64=0.0,
        sum2::Float64=0.0,
        min::Float64=+Inf64,
        max::Float64=-Inf64,
    ) = new(count, sum, sum2, min, max, Float64[])
end

add!(estimator::Estimator, value::Int) = add!(estimator, Float64(value))

function add!(estimator::Estimator, value::Float64)
    estimator.count += 1
    estimator.sum += value
    estimator.sum2 += value^2
    if estimator.min > value
        estimator.min = value
    end
    if estimator.max < value
        estimator.max = value
    end
    push!(estimator.values, value)
end

@inline function mean(estimator::Estimator)
    return estimator.sum / estimator.count
end
@inline function Base.minmax(estimator::Estimator)
    return (estimator.min, estimator.max)
end
std(estimator::Estimator, me::Float64) =
    sqrt(max(0.0, estimator.sum2 / estimator.count - me^2))
std(estimator::Estimator) =
    sqrt(max(0.0, estimator.sum2 / estimator.count - mean(estimator)^2))
median(estimator::Estimator) = Statistics.median(estimator.values)

"""
    (lower,upper)=meanCI(estimator, percent)

Compute confidence interval for the mean interval.
"""
function meanCI(estimator::Estimator, percent::AbstractFloat, mn::AbstractFloat, st::AbstractFloat)
    if estimator.count >= 2
        alpha = 1 - percent
        tStar = quantile(TDist(estimator.count - 1), 1 - alpha / 2)
        l = tStar * st / sqrt(estimator.count)
    else
        l = NaN
    end
    return (mn - l, mn + l)
end
function meanCI(estimator::Estimator, percent::AbstractFloat)
    mn = mean(estimator)
    st = std(estimator, mn)
    return meanCI(estimator, percent, mn, st)
end

"""
    (lower,upper)=range(estimator::Estimator, percent::AbstractFloat)

Compute confidence interval for the values.
"""

function rangeCI(estimator::Estimator, percent::AbstractFloat)
    qt = quantile!(estimator.values, [(1.0 - percent) / 2, (1.0 + percent) / 2])
    return (qt[1], qt[2])
end

function Base.display(estimator::Estimator)
    @printf("min =%12.4f, max=%12.4f, 95%% range=(%12.4f,%12.4f)\n",
        minmax(estimator)..., rangeCI(estimator, 0.95)...)
    mn = mean(estimator)
    st = std(estimator, mn)
    @printf("mean=%12.4f, 95%% ci=(%12.4f,%12.4f)\n",
        mean(estimator), meanCI(estimator, 0.95)...)
    @printf("std =%12.4f\n", st)
end

#####################################
## Linear Regression for 45-def plots
#####################################

# TODO: should have a Plots version

"""
Compute and (optionally) plot 45-degree plots comparing
(scalar) actual values with model forecast.
"""
function actualVSforecast45deg(
    actual::Vector{FloatM},
    forecast::Vector{FloatM},
) where {FloatM}
    K = length(actual)
    @assert length(forecast) == K
    # model 1: actual = forecast + err
    rmse = norm(actual - forecast) / sqrt(K)
    # model 2: actual = a forecast + b + err = [forecast 1] * [a;b] + err
    X = [reshape(forecast, K, 1) ones(FloatM, K, 1)]
    ab = X \ actual
    rmseLin = norm(actual - X * ab) / sqrt(K)
    return (rmse, rmseLin, ab)
end

# FIXME: this function needs to be ported to Plots, rather than PythonPlot
function actualVSforecast45deg!(
    ax,#::PythonCall.Py,
    actual::Vector{FloatM},
    forecast::Vector{FloatM};
    actualLabel::String="actual",
    forecastLabel::String="forecast",
    title::String="",
    mn::FloatM=min(minimum(actual), minimum(forecast)),
    mx::FloatM=max(maximum(actual), maximum(forecast)),
    alpha::FloatM=NaN,
    reverseAxis=false,
) where {FloatM}
    if reverseAxis
        (rmse, rmseLin, ab) = actualVSforecast45deg(actual, forecast)
    else
        (rmse, rmseLin, ab) = actualVSforecast45deg(forecast, actual)
    end
    if isnan(alpha)
        alpha = length(actual) <= 50 ? 1.0 :
                length(actual) <= 100 ? 0.25 : 0.1
    end
    if reverseAxis
        ax.plot(forecast, actual, '.'; alpha)
    else
        ax.plot(actual, forecast, '.'; alpha)
    end
    ax.plot([mn, mx], [mn, mx], "--", color="gray")
    ax.plot([mn, mx], ab[1] .* [mn, mx] .+ ab[2], ":", color="gray")
    ax.set_xlim([mn, mx])
    ax.set_ylim([mn, mx])
    ax.set_aspect("equal")
    ax.grid(true)
    ax.legend([
        "data",
        @sprintf("45° (rmse=%.3f)", rmse),
        @sprintf("fit (rmse=%.3f)", rmseLin),
    ])
    if reverseAxis
        ax.set_xlabel(forecastLabel)
        ax.set_ylabel(actualLabel)
    else
        ax.set_ylabel(forecastLabel)
        ax.set_xlabel(actualLabel)
    end
    ax.set_title(title)
    return (rmse, rmseLin, ab)
end

end