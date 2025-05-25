include("fftutilities.jl")

export spectralSolve

# Assuming for now that dt is fixed
# If custom outputter is not provided, then resort to default
# First step is stored during initilization of output
function spectral_solve(prob::SOP, scheme::SA=MSS3(), output::O=Output(prob, 100); 
    resume::Bool=false) where {SOP<:SpectralODEProblem,SA<:AbstractODEAlgorithm,O<:Output}
    # Initialize cache
    if resume && output.store_hdf && haskey(output.simulation, "cache_backup")
        cache = restore_cache(output.simulation, prob, scheme)
        t = read(output.simulation, "cache_backup/last_t")
        step = read(output.simulation, "cache_backup/last_step")
    else
        cache = get_cache(prob, scheme)
        t = first(prob.tspan)
        step = 0
    end

    # Time step
    dt = prob.dt

    # Calculate number of steps
    total_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / dt)

    try
        # This method assumes step number does not overflow!
        while step < total_steps
            perform_step!(cache, prob, t)

            # Increment step and time #TODO fix time tracking
            step += 1
            t += dt

            handle_output!(output, step, cache.u, prob, t)
            sleep(1e-10000000000000) #To be able to interupt simulation
        end
    catch error
        # Interupt the error, so that the code does not halt
        showerror(stdout, error)
    end

    # Store the cache to be able to resume simulations
    output_cache!(output, cache, step, t)

    # TODO catch edge case

    # Write buffer to file
    output.store_hdf ? flush(output.file) : nothing

    # Returns output struct
    return output
end