module TrackBenchmarks

export Description, saveBenchmark

using Printf
#using Formatting

using CSV
using DataFrames
using Git
using Dates
using TimeZones

# TODO remove from IPsparse

"""
Structure used to store the description of a problem, solver, etc.

# Example
   d=Description("quadratic minmax";linearSolver=:LDL,equalityTolerance=1e-8,muFactorAggressive=.9)
   d=Description(solveTime=.1,solveTimeWithoutPrint=.05)
"""
struct Description
    name
    parNames
    parValues
    function Description(name::String; varargs...)
        parNames = [string(key) for (key, value) in varargs]
        parValues = [value for (key, value) in varargs]
        return new(name, parNames, parValues)
    end
    function Description(; varargs...)
        return Description(""; varargs...)
    end
end

# TODO: pruning not implemented
"""
# Example
   saveBenchmark(
    "IPbenchmarks.csv";
    solver=Description("quadratic minmax",linearSolver=:LDL,equalityTolerance=1e-8,muFactorAggressive=.9),
    problem=Description("Rock paper Scissors",nU=10,nEqU=1),
    time=Description(solveTime=.1,solveTimeWithoutPrint=.05),
    pruneBy=Hour(1))
"""
function saveBenchmark(
    filename::String;
    solver::Description,
    problem::Description,
    time::Description,
    benchmarkTime::ZonedDateTime=now(localzone()),
    pruneBy::Period=Hour(0))

    ## Read previous benchmark
    try
        df = DataFrame(CSV.File(filename))
    catch err
        display(err)
        @printf("saveBenchmark: could not read benchmark file \"%s\"\n", filename)
        df = []
    end

    if !isempty(df)
        # Convert Dates
        #display(df.benchmarkTime)
        #display(df.gitCommitTime)
        df.benchmarkTime = ZonedDateTime.(String.(df.benchmarkTime), "yyyy-mm-dd H:M:S.s z")
        df.gitCommitTime = ZonedDateTime.(String.(df.gitCommitTime), "yyyy-mm-dd H:M:S.s z")
        #display(df.benchmarkTime)
        #display(df.gitCommitTime)

        # Convert solve times
        #@show str1 = replace.(df.timeValues, r"[^[]*\[([^]]*)\]" => s"\1")
        #@show str2 = split.(str1, ",")
        #@show str3 = string.(str2)
        str2vector(str) = parse.(Float64, split(replace(str, r"[^[]*\[([^]]*)\]" => s"\1"), ","))
        df.timeValues = str2vector.(df.timeValues)
    end

    gitCommitHash = ""
    gitCommitTime = ""
    try
        gitCommitHash = readchomp(`$(git()) log -1 --format='%H'`)
        gitCommitTime = readchomp(`$(git()) log -1 --format='%ai'`)
        gitCommitTime = ZonedDateTime(gitCommitTime, "yyyy-mm-dd H:M:S z")
    catch
        @printf("saveBenchmark: could not get git commit\n")
    end
    ## Add current benchmark
    df1 = DataFrame(
        benchmarkTime=[benchmarkTime],
        solverName=String[solver.name],
        problemName=[problem.name],
        problemValues=[string(problem.parValues)],
        timeValues=[time.parValues],
        timesNames=[string(time.parNames)],
        solverValues=[string(solver.parValues)],
        problemParameters=[string(problem.parNames)],
        solverParameters=[string(solver.parNames)],
        gitCommitHash=[gitCommitHash],
        gitCommitTime=[gitCommitTime],)
    #display(df)
    #display(df1)

    save = false

    ## prune
    if isempty(df)
        @printf("saveBenchmark: added to empty file\n")
        df = df1
        save = true
    elseif iszero(pruneBy)
        @printf("saveBenchmark: added (zero pruneBy)\n")
        df = vcat(df, df1)
    else
        ## compute rows that match solver+problem+prune time
        fields2match = [
            3, # problemName
            4, # problemValues
            8, # problemParameters
            6, # timesNames
            2, # solverName
            7, # solverValues
            9, # solverParameters
        ]
        tMatch = (df[:, fields2match] .== df1[:, fields2match])
        kMatchProblem = vec(collect(all(Matrix(tMatch), dims=2))) # convert BitMatrix to Bool
        kMatchTime = (abs.(df.benchmarkTime - benchmarkTime) .< Dates.CompoundPeriod(pruneBy))
        kMatch = kMatchProblem .& kMatchTime

        if !any(kMatch)
            @printf("saveBenchmark: added (no match)\n")
            df = vcat(df, df1)
            save = true
        else
            dfMatch = copy(df[kMatch, :])
            ## find rows with all times better or equal to current times
            solveTimes2Matrix = hcat(dfMatch.timeValues...)'
            solveTimeAsVector = hcat(df1.timeValues[1]...)
            kBetterTime = all(solveTimes2Matrix .<= solveTimeAsVector, dims=2)
            if !any(kBetterTime)
                @printf("saveBenchmark: added (better times than matched)\n")
                df = vcat(df, df1)
                save = true
            end
        end
    end

    if save
        ## save updates file
        # making times more readable
        dfs = copy(df)
        dfs.benchmarkTime = Dates.format.(df.benchmarkTime, "yyyy-mm-dd H:M:S.s z")
        dfs.gitCommitTime = Dates.format.(df.gitCommitTime, "yyyy-mm-dd H:M:S.s z")

        @printf("saveBenchmark: saving \"%s\"\n", filename)
        CSV.write(filename, dfs)
    end
    return df
end


end