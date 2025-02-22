using HDF5
using H5Zblosc
using Dates

mutable struct Output
    stride::Number
    diagnostics::AbstractArray
    u::AbstractArray
    t::AbstractArray
    filename::AbstractString
    file::HDF5.File
    simulation::HDF5.Group
    h5_kwargs::NamedTuple

    function Output(prob, N_data::Integer, diagnostics::AbstractArray=DEFAULT_DIAGNOSTICS,
        filename::AbstractString=basename(tempname())*".h5"; h5_kwargs...)

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
        stride = floor(Int, N_steps / (N_data - 1))

        # Calculate number of samples with rounded sampling rate
        N = floor(Int, N_steps / stride) + 1

        if N != N_data
            N_data = N
            @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        end

        # Create HDF5 file
        file = h5open(filename, "cw")

        # Create new simulation group
        simulation = create_group(file, "$(now())") # Perhaps should be named after parameters instead TODO decide

        if isempty(h5_kwargs)
            h5_kwargs = (blosc=3,)
        end

        # Create dataset for fields and time
        dset = create_dataset(simulation, "fields", datatype(Float64), (size(prob.u0)..., typemax(Int64)),
            chunk=(size(prob.u0)..., 1); h5_kwargs...)
        HDF5.set_extent_dims(dset, (size(prob.u0)..., N_data))
        dset = create_dataset(simulation, "t", datatype(Float64), (typemax(Int64),),
            chunk=(1,); h5_kwargs...)
        HDF5.set_extent_dims(dset, (N_data,))

        # TODO something with parameters
        # perhaps something like: create_attribute(simulation, prob) and create new method

        # Allocate data for fields
        # TODO maybe remove u?
        u = Vector{typeof(prob.u0)}(undef, N_data)
        u[1] = prob.u0
        simulation["fields"][:,:,:,1] = u[1]#TODO remove hard coding 
        t = zeros(N_data)
        t[1] = first(prob.tspan)
        simulation["t"][1] = t[1]

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            initialize_diagnostic!(diagnostic, prob, simulation, h5_kwargs)
        end

        # Create output
        new(stride, diagnostics, u, t, filename, file, simulation, h5_kwargs)
    end
end

function handle_output!(output::Output, step::Integer, u::AbstractArray, prob::SpectralODEProblem, t::Number)
    if step % output.stride == 0
        # TODO move logic to spectralSolve
        #if prob.domain.Nx % 2 == 0
        #    u[:, prob.domain.Nx÷2+1, :] .= 0
        #end
        #if prob.domain.Ny % 2 == 0
        #    u[prob.domain.Ny÷2+1, :, :] .= 0
        #end

        # TODO implement HDF5
        # TODO add method to recover field
        output.u[step÷output.stride+1] = real(transform(u, prob.domain.transform.iFT))
        output.t[step÷output.stride+1] = t
    end

    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.sampleStep == 0
            perform_diagnostic!(diagnostic, step, u, prob, t) #diagnostic.data[step÷diagnostic.sampleStep] = diagnostic.method(U, prob, t)
        end
    end

    # Check if first value is NaN, if the matrix has one NaN the whole array will turn NaN after fft
    if isnan(u[1])
        error("Breakdown occured at t=$t")
    end
end

# Gives a more managle array compared to output.u
function extract_output(output::Output)
    # TODO add check for data type of vector
    Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))
end

# TODO implement
function extract_diagnostic(data::Vector)
    Array(reshape(reduce(hcat, data), size(data[1])..., length(data)))
end
# These are just stack() ^^