include("fftutilities.jl")

export spectralSolve

# Assuming for now that dt is fixed
# If custom outputter is not provided, then resort to default
# First step is stored during initilization of output
function spectral_solve(prob::SpectralODEProblem, scheme::AbstractODEAlgorithm=MSS3(), 
                        output::Output=Output(prob, 100); resume=false)
    # Initialize cache
    if resume
        cache = get_cache(prob, scheme)
        dt = prob.dt
        t = first(prob.tspan)
        step = 0
    else
        cache = get_cache(prob, scheme)
        dt = prob.dt
        t = first(prob.tspan)
        step = 0
    end
    
    # Calculate number of steps
    total_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / dt)

    try
        # This method assumes step number does not overflow!
        while step < total_steps
            perform_step!(cache, prob, t)

            # Increment step and time 
            step += 1
            # TODO add time tracking to perform_step?
            t += dt

            handle_output!(output, step, cache.u, prob, t)
            sleep(1e-10000000000000) #To be able to interupt simulation
        end
    catch error
        # Interupt the error, so that the code does not halt
        showerror(stdout,error)
    end
    
    # Store the cache to be able to resume simulations
    output_cache!(output, cache, step, t)
    
    # TODO catch edge case and close file
    
    #if false #TODO add output.email
    #    send_mail("Simulation finished!")
    #end

    # Returns output struct
    return output
end