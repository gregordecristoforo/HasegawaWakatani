#include("spectralODEProblem.jl")
include("schemes.jl")
include("fftutilities.jl")

export spectralSolve

# Assuming for now that dt is fixed
# If custom outputter is not provided, then resort to default
# First step is stored during initilization of output
function spectral_solve(prob::SpectralODEProblem, scheme=MSS3(), output=Output(prob, 100))
    # Initialize cache
    cache = get_cache(prob, scheme)

    # Auxilary variables
    dt = prob.dt
    t = first(prob.tspan) + dt
    step = 1

    # Calculate number of steps
    total_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / dt) + 1

    # This method assumes step number does not overflow!
    while step < total_steps
        perform_step!(cache, prob, t)
        handle_output!(output, step, cache.u, prob, t)

        # Increment step and time 
        step += 1
        t += dt
    end

    # TODO catch edge case

    # Returns output struct
    return output
end