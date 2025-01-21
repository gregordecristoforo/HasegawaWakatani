#include("spectralODEProblem.jl")
include("schemes.jl")
include("fftutilities.jl")

export spectralSolve

# If custom outputter is not provided, then resort to default
function spectral_solve(prob::SpectralODEProblem, scheme=MSS3(), output=Output(prob, 100))
    # Initialize cache
    cache = get_cache(prob, scheme)

    # Auxilary variables
    t = first(prob.tspan)
    tend = last(prob.tspan)
    dt = prob.dt
    domain = prob.domain
    step = 1

    while t <= tend
        perform_step!(cache, prob, t)
        handleOutput!(output, step, cache.u, prob, t)

        # Increment step and time 
        step += 1
        t += dt
    end

    # Returns output struct
    return output
end