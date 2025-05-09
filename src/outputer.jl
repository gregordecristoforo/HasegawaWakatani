using HDF5
using H5Zblosc
using Dates

mutable struct Output{DV<:AbstractArray,U<:AbstractArray,UB<:AbstractArray,T<:AbstractArray,
    FN<:AbstractString,F<:Union{HDF5.File,Nothing},S<:Union{HDF5.Group,Nothing},PT<:Function,
    K<:Any} #TODO figure out type of K

    stride::Int
    diagnostics::DV
    u::U
    U_buffer::UB
    #v::V
    t::T
    filename::FN
    file::F
    simulation::S
    physical_transform::PT
    store_hdf::Bool
    store_locally::Bool
    h5_kwargs::K #Possibly also called a filter

    function Output(prob::SOP, N_data::Int, diagnostics::DV=DEFAULT_DIAGNOSTICS,
        filename::FN=basename(tempname()) * ".h5"; physical_transform::PT=identity,
        simulation_name::SN=:timestamp, store_hdf::Bool=true, store_locally::Bool=true,
        h5_kwargs...) where {SOP<:SpectralODEProblem,DV<:AbstractArray,FN<:AbstractString,
        PT<:Function,SN<:Union{AbstractString,Symbol}}

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

        # Check if compatible
        if N != N_data
            N_data = N
            @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        end

        # Copy initial condition and possibly transform
        U = copy(prob.u0)
        physical_transform(U)

        # Store in hdf5 format if user wants
        if store_hdf
            # Check for filename extension
            if splitext(filename)[end] == ""
                filename = splitext(filename)[1] * ".h5"
            end

            # Create HDF5 file
            file = h5open(filename, "cw")

            # Handle simulation_name
            if simulation_name == :timestamp
                simulation_name = "$(now())"
            elseif simulation_name == :parameters
                simulation_name = parameter_string(prob.p)
            elseif simulation_name isa String
                nothing
            else
                error("$simulation_name is not a valid input")
            end

            # Make sure simulation group does not allready exist
            if !haskey(file, simulation_name)
                simulation = create_group(file, simulation_name)

                # Default h5 options
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

                # Store the initial conditions
                simulation["fields"][fill(:, ndims(U))..., 1] = U
                simulation["t"][1] = first(prob.tspan)

                # Store attributes
                write_attribute(simulation, prob)
            else
                simulation = open_group(file, simulation_name)
                # TODO add possibility to expand length of fields and t
            end
        else
            file = nothing
            filename = ""
            simulation = nothing
        end

        # Store "locally" as in memory
        if store_locally
            # Allocate local data for fields
            u = [zero(prob.u0) for _ in 1:N_data]
            u[1] .= U
            t = zeros(N_data)
            t[1] = first(prob.tspan)
        else
            # Do not allocate data for fields locally
            u = []
            t = []
        end

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            initialize_diagnostic!(diagnostic, U, prob, simulation, h5_kwargs,
                store_hdf=store_hdf, store_locally=store_locally)
        end

        # Create output
        new{typeof(diagnostics),typeof(u),typeof(U),typeof(t),typeof(filename),typeof(file),
            typeof(simulation),typeof(physical_transform),typeof(h5_kwargs)}(stride,
            diagnostics, u, U, t, filename, file, simulation, physical_transform,
            store_hdf, store_locally, h5_kwargs)
    end
end

function handle_output!(output::O, step::Int, u::T, prob::SOP, t::N) where {O<:Output,
    T<:AbstractArray,SOP<:SpectralODEProblem,N<:Number}

    # Keeps track such that fields only transformed once
    transformed = false

    # Remove modes using user defined function
    prob.remove_modes(u, prob.domain)

    # Auxilary name
    U = output.U_buffer
    
    # Check wether or not to output 
    if step % output.stride == 0
        # Transform data
        spectral_transform!(U, u, prob.domain.transform.iFT)
        output.physical_transform(U)
        transformed = true

        # Calculate index
        idx = step ÷ output.stride + 1

        # Store in hdf
        if output.store_hdf
            output.simulation["fields"][fill(:, ndims(u))..., idx] = U
            output.simulation["t"][idx] = t
        end

        # TODO perhaps a error should be thrown somewhere if there is no output?
        if output.store_locally
            output.u[idx] .= U
            output.t[idx] = t
        end
    end

    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.sampleStep == 0
            # Check if diagnostic assumes physical field and transform if not yet done
            if !diagnostic.assumesSpectralField && !transformed
                # Transform data
                spectral_transform!(U, u, prob.domain.transform.iFT)
                output.physical_transform(U)
                transformed = true
            end

            # Passes the logic onto perform_diagnostic! to do diagnostic and store data
            if diagnostic.assumesSpectralField
                perform_diagnostic!(diagnostic, step, u, prob, t,
                    store_hdf=output.store_hdf, store_locally=output.store_locally)
            else
                perform_diagnostic!(diagnostic, step, U, prob, t,
                    store_hdf=output.store_hdf, store_locally=output.store_locally)
            end
        end
    end

    # TODO remove?
    if step % 1000 == 0
        output.store_hdf ? flush(output.file) : nothing 
    end

    # Check if last value is NaN, if the matrix has one NaN the whole array will turn NaN after fft
    if isnan(u[end])
        error("Breakdown occured at t=$t")
    end
end

# Perhaps one could look into HDF5 compound types in the future
function output_cache!(output::O, cache::C, step::Int, t::N) where {O<:Output,
    C<:AbstractCache,N<:Number}
    if output.store_hdf
        # Create or open a h5group for the backup cache
        if haskey(output.simulation, "cache_backup")
            cache_group = open_group(output.simulation, "cache_backup")
        else
            cache_group = create_group(output.simulation, "cache_backup")
        end

        # Get all the attributes of cache
        keys = fieldnames(typeof(cache))

        # Dump cache
        for key in keys
            val = getfield(cache, key)
            # Do not want to bother with storing Tableau, easy to recover
            if !isa(val, AbstractTableau)
                if haskey(cache_group, string(key))
                    write(cache_group[string(key)], val)
                else
                    cache_group[string(key)] = val
                end
            end
        end

        # Backup last step and time
        if haskey(cache_group, "last_t") #Assumes if one exist the other does as well
            write(cache_group["last_t"], t)
            write(cache_group["last_step"], step)
        else
            cache_group["last_t"] = t
            cache_group["last_step"] = step
        end
    end
end

function restore_cache(simulation::HDF5.Group, prob::SOP, scheme::SA) where {
    SOP<:SpectralODEProblem,SA<:AbstractODEAlgorithm}

    # Create cache container and get the fieldnames
    cache = get_cache(prob, scheme)
    keys = fieldnames(typeof(cache))

    # Restore cache
    for key in keys
        val = getfield(cache, key)
        # Skip restoring Tableau
        if !isa(val, AbstractTableau)
            # Restore cache
            setproperty!(cache, key, read(simulation["cache_backup"], string(key)))
        end
    end

    return cache
end

import HDF5.write_attribute
function write_attribute(simulation::HDF5.Group, prob::SOP) where {SOP<:SpectralODEProblem}
    write_attribute(simulation, "dt", prob.dt)
    write_attribute(simulation, "dx", prob.domain.dx)
    write_attribute(simulation, "dy", prob.domain.dy)
    write_attribute(simulation, "Lx", prob.domain.Lx)
    write_attribute(simulation, "Ly", prob.domain.Ly)
    write_attribute(simulation, "Nx", prob.domain.Nx)
    write_attribute(simulation, "Ny", prob.domain.Ny)
    write_attribute(simulation, "anti_aliased", prob.domain.anti_aliased)
    write_attribute(simulation, "realTransform", prob.domain.realTransform)
    for (key, val) in prob.p
        write_attribute(simulation, key, val)
    end
end

function parameter_string(parameters::P) where {P<:Dict}
    tmp = [string(key, "=", value) for (key, value) in sort(collect(parameters))]
    join(tmp, ", ")
end

#------------------------------ Removal of modes -------------------------------------------
# TODO perhaps moved to utilitise

function remove_zonal_modes!(u::U, d::D) where {U<:AbstractArray,D<:Domain}
    @inbounds u[1, :, :] .= 0
end

function remove_streamer_modes!(u::U, d::D) where {U<:AbstractArray,D<:Domain}
    @inbounds u[:, 1, :] .= 0
end

function remove_asymmetric_modes!(u::U, domain::D) where {U<:AbstractArray,D<:Domain}
    if domain.Nx % 2 == 0
        @inbounds u[:, domain.Nx÷2+1, :] .= 0
    end
    if Ny % 2 == 0
        @inbounds u[domain.Ny÷2+1, :, :] .= 0
    end
end

function remove_nothing(u::U, d::D) where {U<:AbstractArray,D<:Domain}
    nothing
end

# -------------------------------------- Old -----------------------------------------------

# Gives a more managable array compared to output.u
function extract_output(output::Output)
    # TODO add check for data type of vector
    Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))
end

function extract_diagnostic(data::Vector)
    Array(reshape(reduce(hcat, data), size(data[1])..., length(data)))
end
# These are just stack() ^^

# function remove_zonal_modes!(u::U) where {U<:AbstractArray}
#     @inbounds u[1, ntuple(_ -> :, ndims(u) - 1)...] .= 0
# end