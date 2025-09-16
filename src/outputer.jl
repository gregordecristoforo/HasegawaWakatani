# ------------------------------------- Outputer -------------------------------------------

mutable struct Output{DV<:AbstractArray{<:Diagnostic},U<:AbstractArray,UB<:AbstractArray,
    T<:AbstractArray,F<:Union{HDF5.File,Nothing},S<:Union{HDF5.Group,Nothing},PT<:Function,
    K<:Any} #TODO figure out type of K

    stride::Int
    diagnostics::DV
    u::U
    U_buffer::UB
    #v::V
    t::T
    file::F                                 # TODO remove as it is in output.simulation.file
    simulation::S
    physical_transform::PT
    store_hdf::Bool
    store_locally::Bool
    h5_kwargs::K #Possibly also called a filter

    function Output(prob::SOP, N_data::Integer, diagnostics::DV=DEFAULT_DIAGNOSTICS,
        filename::FN=basename(tempname()) * ".h5"; physical_transform::PT=identity,
        simulation_name::SN=:timestamp, store_hdf::Bool=true, store_locally::Bool=true,
        h5_kwargs...) where {SOP<:SpectralODEProblem,DV<:AbstractArray,FN<:AbstractString,
        PT<:Function,SN<:Union{AbstractString,Symbol}}

        # Compute number of samples to be stored and stride distance
        N_data, stride = prepare_sampling_coverage(prob, N_data)

        # Prepare initial state
        u0, t0 = prepare_initial_state(prob, physical_transform=physical_transform)

        # Merge h5_kwargs with default kwargs
        h5_kwargs = merge((blosc=3,), h5_kwargs)

        # Setup HDF5 storage if wanted
        file, simulation = setup_hdf5_storage(filename, simulation_name, prob, u0, t0, N_data;
            store_hdf=store_hdf, h5_kwargs=h5_kwargs)

        # Setup local (in memory) storage if wanted
        u, t = setup_local_storage(u0, t0, N_data; store_locally=store_locally)

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            # TODO fix arguments
            initialize_diagnostic!(diagnostic, prob, u0, t0, simulation, h5_kwargs,
                store_hdf=store_hdf, store_locally=store_locally)
        end

        # Create output
        new{typeof(diagnostics),typeof(u),typeof(u0),typeof(t),typeof(file),
            typeof(simulation),typeof(physical_transform),typeof(h5_kwargs)}(stride,
            diagnostics, u, u0, t, file, simulation, physical_transform,
            store_hdf, store_locally, h5_kwargs)
    end
end

# ----------------------------------- Helpers ----------------------------------------------

function prepare_sampling_coverage(prob, N_data)
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

    return N_data, stride
end

"""
"""
function prepare_initial_state(prob; physical_transform=identity)
    u0 = physical_transform(copy(prob.u0))
    t0 = first(prob.tspan)
    return u0, t0
end

"""
"""
function setup_hdf5_storage(filename, simulation_name, prob, u0, t0, N_data;
    store_hdf=store_hdf, h5_kwargs=h5_kwargs)

    if store_hdf
        filename = add_h5_if_missing(filename)

        # Create HDF5 file
        file = h5open(filename, "cw")

        # Create simulation name
        simulation_name = handle_simulation_name(simulation_name, prob)

        # Check how to handle simulation group
        if !haskey(file, simulation_name)
            simulation = setup_simulation_group(file, simulation_name, prob, u0, t0, N_data;
                h5_kwargs)
        else
            simulation = open_group(file, simulation_name)
        end

        return file, simulation
    else
        return nothing, nothing
    end
end

"""
    add_h5_if_missing(filename)
  Makes sure the filename has an extension, if not the `.h5` extension is added. 
"""
function add_h5_if_missing(filename)
    splitext(filename)[end] == "" ? splitext(filename)[1] * ".h5" : filename
end

"""
    handle_simulation_name(simulation_name)
  Creates a `simulation_name` string based on the users input. 
  ### Supported symbols:
  * `:timestamp` creates a timestamp string using `Dates.now()`.
  * `:parameters` creates a string with the parameter names and values.
"""
function handle_simulation_name(simulation_name, prob)
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

    return simulation_name
end

"""
"""
function setup_simulation_group(file, simulation_name, prob, u0, t0, N_data; h5_kwargs)

    # Create simulation group
    simulation = create_group(file, simulation_name)

    # Create dataset for fields and time
    dset = create_dataset(simulation, "fields", datatype(eltype(u0)),
        (size(u0)..., typemax(Int64)), chunk=(size(u0)..., 1); h5_kwargs...)
    HDF5.set_extent_dims(dset, (size(u0)..., N_data))
    dset = create_dataset(simulation, "t", datatype(eltype(t0)),
        (typemax(Int64),), chunk=(1,); h5_kwargs...)
    HDF5.set_extent_dims(dset, (N_data,))

    # Store the initial conditions
    write_state(simulation, u0, t0)

    # Store attributes
    write_attributes(simulation, prob)

    return simulation
end

"""
    write_state(simulation, u, t)
  Writes the state `u` at time `t` to the simulation group `simulation`.
"""
function write_state(simulation, u, t)
    simulation["fields"][fill(:, ndims(u))..., 1] = u #TODO possibly problem is here
    simulation["t"][1] = t
end

"""
    write_attributes(simulation, prob::SpectralODEProblem)
    write_attributes(simulation, domain::AbstractDomain)
  Writes the esential attributes of the container to the simulation group `simulation`. The 
  `SpectralODEProblem` also writes the `domain` properties.
"""
function write_attributes(simulation, prob::SpectralODEProblem)
    write_attribute(simulation, "dt", prob.dt)
    write_attributes(simulation, prob.domain)
    # TODO add multiple dispatch to this
    for (key, val) in pairs(prob.p)
        write_attribute(simulation, string(key), val)
    end
end

function write_attributes(simulation, domain::AbstractDomain)
    # Construct list of attributes by removing derived attributes
    attributes = setdiff(fieldnames(typeof(domain)), (:x, :y, :kx, :ky, :SC, :transform, :precision))
    for attribute in attributes
        write_attribute(simulation, string(attribute), getproperty(domain, attribute))
    end
end

"""
    setup_local_storage(u0, t0, N_data; store_locally=store_locally)
  Allocates vectors in memory for storing the fields alongside the time if the user wants it,
  otherwise empty vectors are returned.
"""
function setup_local_storage(u0, t0, N_data; store_locally=store_locally)

    if store_locally
        # Allocate local data for fields
        u = [zero(u0) for _ in 1:N_data]
        u[1] .= u0
        t = zeros(N_data)
        t[1] = t0
    else
        u, t = [], []
    end

    return u, t
end

# TODO fix show method

# ----------------------------------- Handlers ---------------------------------------------

function handle_output!(output::O, step::Integer, u::T, prob::SOP, t::N) where {O<:Output,
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

    # Check if first value is NaN, if one value is NaN the whole Array will turn NaN after FFT
    assert_no_nan(u)
end

# Perhaps one could look into HDF5 compound types in the future
function output_cache!(output::O, cache::C, step::Integer, t::N) where {O<:Output,
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
                if isa(val, CuArray)
                    val = Array(val) # Download to CPU
                end

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

    # TODO move this logic to resuming simulations
    # Expand length of fields and t
    #HDF5.set_extent_dims(simulation["fields"], (size(prob.u0)..., N_data))
    #HDF5.set_extent_dims(simulation["t"], (N_data,))

    return cache
end

# -------------------------------------- Utilities -----------------------------------------

function parameter_string(parameters::P) where {P<:AbstractDict}
    tmp = [string(key, "=", value) for (key, value) in sort(collect(parameters))]
    join(tmp, ", ")
end

function parameter_string(parameters::P) where {P<:NamedTuple}
    tmp = [string(key, "=", value) for (key, value) in sort(collect(pairs(parameters)))]
    join(tmp, ", ")
end

function assert_no_nan(u::AbstractArray)
    if isnan(u[1])
        error("Breakdown occured at t=$t")
    end
end

function assert_no_nan(u::CuArray)
    if CUDA.@allowscalar isnan(u[1])
        error("Breakdown occured at t=$t")
    end
end

#------------------------------ Removal of modes -------------------------------------------
# TODO perhaps moved to utilitise

function remove_zonal_modes!(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    @inbounds u[1, :, :] .= 0
end

function remove_streamer_modes!(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    @inbounds u[:, 1, :] .= 0
end

function remove_asymmetric_modes!(u::U, domain::D) where {U<:AbstractArray,
    D<:AbstractDomain}
    if domain.Nx % 2 == 0
        @inbounds u[:, domain.Nx÷2+1, :] .= 0
    end
    if Ny % 2 == 0
        @inbounds u[domain.Ny÷2+1, :, :] .= 0
    end
end

function remove_nothing(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    nothing
end

# -------------------------------------- Old -----------------------------------------------

# Gives a more managable array compared to output.u
function extract_output(output::Output)
    Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))
end

function extract_diagnostic(data::Vector)
    Array(reshape(reduce(hcat, data), size(data[1])..., length(data)))
end
# These are just stack() ^^