module EstimationTools

using Reexport

include(raw"Estimators.jl")
@reexport using .Estimators

include(raw"LeastSquaresBatch.jl")
@reexport using .LeastSquaresBatch

include(raw"LeastSquaresIncremental.jl")
@reexport using .LeastSquaresIncremental

end # module EstimationTools
