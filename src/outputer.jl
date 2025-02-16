
mutable struct Output
    fieldStep::Number
    diagnostics::AbstractArray
    u::AbstractArray
    t::AbstractArray
    filename::Any

    function Output(prob, N_data::Integer, diagnostics=DEFAULT_DIAGNOSTICS, filename=nothing) #TODO auto filename

        # Calculate number of total samples
        N_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / prob.dt)

        # Bound checking
        if N_data == -1
            N_data = N_steps + 1
        elseif N_data <= 1
            error("N_data must be greater than 2, or use -1 to get all the data")
        elseif N_data > N_steps
            N_data = N_steps + 1
            @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        end

        # Calculate number of evolution steps between samples
        fieldStep = floor(Int, N_steps / (N_data - 1))

        # Calculate number of samples with rounded sampling rate
        N = floor(Int, N_steps / fieldStep) + 1

        if N != N_data
            N_data = N
            @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        end

        # Allocate data for fields
        u = Vector{typeof(prob.u0)}(undef, N_data)
        u[1] = prob.u0
        t = zeros(N_data)
        t[1] = first(prob.tspan)

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            initialize_diagnostic!(diagnostic, prob)
        end

        # Create output
        new(fieldStep, diagnostics, u, t, filename)
    end
end

function handle_output!(output::Output, step::Integer, u::AbstractArray, prob::SpectralODEProblem, t::Number)
    if step % output.fieldStep == 0
        # TODO move logic to spectralSolve
        #if prob.domain.Nx % 2 == 0
        #    u[:, prob.domain.Nx÷2+1, :] .= 0
        #end
        #if prob.domain.Ny % 2 == 0
        #    u[prob.domain.Ny÷2+1, :, :] .= 0
        #end

        # TODO implement HDF5
        # TODO add method to recover field
        output.u[step÷output.fieldStep+1] = real(transform(u, prob.domain.transform.iFT))
        output.t[step÷output.fieldStep+1] = t
    end

    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.sampleStep == 0
            perform_diagnostic!(diagnostic, step, u, prob, t) #diagnostic.data[step÷diagnostic.sampleStep] = diagnostic.method(U, prob, t)
        end
    end
end

# Gives a more managle array compared to output.u
function extract_output(output::Output)
    # TODO add check for data type of vector
    Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))
end

# TODO implement
function extract_diagnostic()
end