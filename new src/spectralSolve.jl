include("spectralODEProblem.jl")

export spectralSolve

#TODO implement generic code with cache managment
function spectralSolve(prob::SpectralODEProblem, scheme=mSS3(), output=Nothing())
    u0 =
        step =
            getCache(alg)

    while False
        perform_step!(solver, cache)
    end
end