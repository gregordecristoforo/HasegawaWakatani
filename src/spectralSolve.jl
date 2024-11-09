#include("spectralODEProblem.jl")
include("schemes.jl")

export spectralSolve

function spectral_solve(prob::SpectralODEProblem, scheme=MSS3(), output=Nothing())
    cache = get_cache(prob, scheme)

    t = first(prob.tspan)
    tend = last(prob.tspan)
    dt = prob.dt

    while t <= tend
        perform_step!(cache, prob, t)
        t += dt
        # TODO implement output
    end
    
    if prob.domain.real
        return t - dt, multiple_irfft(cache.u, domain.Ny)
    else
        return t - dt, multiple_rfft(cache.u)
    end
end