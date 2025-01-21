mutable struct Output
    fieldStep::Number
    diagnostics::AbstractArray
    u::AbstractArray
    filename::Any

    # TODO make it more clear?
    # Atm u[:,:,1] = n, u[:,:,2] = Ω
    function Output(prob, fieldStep, diagnostics=default_diagnostics, filename=nothing) #TODO auto filename
        # Extract values
        tend = last(prob.tspan)
        dt = prob.dt

        # Allocate data for fields
        N = floor(Int, tend / dt / fieldStep)
        u = Vector{typeof(prob.u0)}(undef, N)
        u[1] = prob.u0

        # Allocate data for diagnostics... TODO implement
        for diagnostic in diagnostics
            initializeDiagnostic!(diagnostic, prob)
        end

        new(fieldStep, diagnostics, u, filename)
    end
end

#TODO Implement struct Diagnostic, so that one can more easily choose diagnostics to use
# and Diagnostic also includes stuff relating to HDF5

function handleOutput!(output::Output, step::Integer, u::AbstractArray, prob::SpectralODEProblem, t::Number)
    if step % output.fieldStep == 0
        # TODO implement
        #u[prob.domain.Ny÷2+1, :, :] .= 0
        #u[:, prob.domain.Nx÷2+1, :] .= 0

        output.u[step÷output.fieldStep] = real(multi_ifft(u, prob.domain.transform))
    end

    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.sampleStep == 0
            U = irfft(u, prob.domain.Ny) # transform to realspace
            diagnostic.data[step÷diagnostic.sampleStep] = diagnostic.method(U, prob, t)
        end
    end
end

# Gives a more managle array compared to output.u
function extractOutput(output::Output)
    # TODO add check for data type of vector
    Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))
end