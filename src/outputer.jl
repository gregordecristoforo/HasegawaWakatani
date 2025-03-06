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
    h5_kwargs::NamedTuple #Possibly also called a filter

    function Output(prob, N_data::Integer, diagnostics::AbstractArray=DEFAULT_DIAGNOSTICS,
        filename::AbstractString=basename(tempname()) * ".h5"; h5_kwargs...)

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

        #TODO check if .h5 and add it if no extension

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

        # Allocate data for fields
        # TODO maybe remove u?
        u = Vector{typeof(prob.u0)}(undef, N_data)
        U = copy(prob.u0)
        prob.recover_fields!(U)
        u[1] = U
        simulation["fields"][fill(:, ndims(U))..., 1] = u[1]
        t = zeros(N_data)
        t[1] = first(prob.tspan)
        simulation["t"][1] = t[1]

        # Store attributes
        # TODO perhaps something like: write_attribute(simulation, prob) and create new method
        write_attribute(simulation, "dt", prob.dt)
        write_attribute(simulation, "dx", prob.domain.dx)
        write_attribute(simulation, "dy", prob.domain.dy)
        write_attribute(simulation, "L_x", prob.domain.Lx)
        write_attribute(simulation, "L_y", prob.domain.Ly)
        write_attribute(simulation, "N_x", prob.domain.Nx)
        write_attribute(simulation, "N_y", prob.domain.Ny)
        write_attribute(simulation, "anti_aliased", prob.domain.anti_aliased)
        write_attribute(simulation, "real_transform", prob.domain.realTransform)
        for (key, val) in prob.p
            write_attribute(simulation, key, val)
        end

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            initialize_diagnostic!(diagnostic, prob, simulation, h5_kwargs)
        end

        # Create output
        new(stride, diagnostics, u, t, filename, file, simulation, h5_kwargs)
    end
end

function handle_output!(output::Output, step::Integer, u::AbstractArray, prob::SpectralODEProblem, t::Number)
    #remove_zonal_modes!(u)
    
    if step % output.stride == 0
        # TODO move logic to spectralSolve
        #if prob.domain.Nx % 2 == 0
        #    u[:, prob.domain.Nx÷2+1, :] .= 0
        #end
        #if prob.domain.Ny % 2 == 0
        #    u[prob.domain.Ny÷2+1, :, :] .= 0
        #end
        #u[prob.domain.Ny÷2+1, :, :] .= 0

        U = real(transform(u, prob.domain.transform.iFT))
        prob.recover_fields!(U)
        idx = step ÷ output.stride + 1

        output.simulation["fields"][fill(:, ndims(u))..., idx] = U
        output.simulation["t"][idx] = t

        # TODO add method to recover field
        output.u[idx] = U
        output.t[idx] = t
    end

    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.sampleStep == 0
            perform_diagnostic!(diagnostic, step, u, prob, t)
        end
    end

    # Check if last value is NaN, if the matrix has one NaN the whole array will turn NaN after fft
    if isnan(u[end])
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

function remove_zonal_modes!(u::AbstractArray{<:Number})
    @inbounds u[1, ntuple(_ -> :, ndims(u) - 1)...] .= 0
end