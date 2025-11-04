# ------------------------------------------------------------------------------------------
#                                         Outputer                                          
# ------------------------------------------------------------------------------------------

"""
    Output{DV, U, UB, T, S, PT, K}

  A mutable struct for managing simulation output, diagnostics, and storage.
  
  # Fields
  - `stride::Int`: Number of steps between stored samples.
  - `diagnostics::DV`: Array of `Diagnostic`s.
  - `u::U`: Array for storing simulation states in memory.
  - `U_buffer::UB`: Buffer for storing physical space state.
  - `t::T`: Array for storing time points.
  - `simulation::S`: HDF5 group or `Nothing`, representing the simulation storage.
  - `physical_transform::PT`: Function to transform the state in physical space.
  - `store_hdf::Bool`: Whether to store output in HDF5 format.
  - `store_locally::Bool`: Whether to store output in memory.
  - `transformed::Bool`: Indicates if the state has been transformed to physical space.
  - `h5_kwargs::K`: Named tuple of keyword arguments for HDF5 storage.
  
  # Constructor
  
    Output(prob::SOP; filename::FN, diagnostics::DV, stride::Integer, \
    physical_transform::PT, simulation_name::SN, store_hdf::Bool, store_locally::Bool, \
    field_storage_limit::AbstractString, h5_kwargs...)

  Creates an `Output` object for a given `SpectralODEProblem`` `prob`. Handles setup for \
  HDF5 and local storage, initializes diagnostics, and manages sampling stride.
  
  ## Keyword Arguments
  - `filename`: Name of the HDF5 file for output (default: random temporary name).
  - `diagnostics`: Array of diagnostics to compute (default: `DEFAULT_DIAGNOSTICS`).
  - `stride`: Number of steps between samples (default: -1 (lets the program decide)).
  - `physical_transform`: Function to transform state in physical space (default: `identity`).
  - `simulation_name`: Name for the simulation group in HDF5 (default: `:timestamp`).
  - `store_hdf`: Store output in HDF5 file (default: `true`).
  - `store_locally`: Store output in memory (default: `true`).
  - `field_storage_limit`: Limit for field storage (default: empty string).
  - `h5_kwargs...`: Additional keyword arguments for HDF5 storage (merged with defaults).

  # Usage

  Create an `Output` object to manage simulation results, diagnostics, and storage options 
  for a given spectral ODE problem.
"""
mutable struct Output{DV<:AbstractArray{<:Diagnostic},U<:AbstractArray,UB<:AbstractArray,
                      T<:AbstractArray,S<:Union{HDF5.Group,Nothing},PT<:Function,
                      K<:NamedTuple}
    diagnostics::DV
    u::U
    U_buffer::UB
    t::T
    simulation::S
    physical_transform::PT
    store_hdf::Bool
    store_locally::Bool
    transformed::Bool
    h5_kwargs::K #Possibly also called a filter

    function Output(prob::SOP; filename::FN=basename(tempname()) * ".h5",
                    diagnostics::DV=DEFAULT_DIAGNOSTICS,
                    physical_transform::PT=identity,
                    simulation_name::SN=:timestamp,
                    store_hdf::Bool=true,
                    store_locally::Bool=true,
                    storage_limit::AbstractString="",
                    h5_kwargs...) where {SOP<:SpectralODEProblem,DV<:AbstractArray,
                                         FN<:AbstractString,PT<:Function,
                                         SN<:Union{AbstractString,Symbol}}

        # Prepare initial state
        u0, t0 = prepare_initial_state(prob; physical_transform=physical_transform)

        # Merge h5_kwargs with default kwargs
        h5_kwargs = merge((blosc=3,), h5_kwargs)

        # Setup HDF5 storage if wanted
        simulation = setup_hdf5_storage(filename, simulation_name, u0, prob, t0;
                                        store_hdf=store_hdf, h5_kwargs=h5_kwargs)

        # Setup local (in memory) storage if wanted
        u, t = setup_local_storage(u0, t0; store_locally=false) # Currently disabled TODO re-enable

        # Allocate data for diagnostics
        for diagnostic in diagnostics
            initialize_diagnostic!(diagnostic, simulation, h5_kwargs, u0, prob, t0;
                                   store_hdf=store_hdf, store_locally=store_locally)
        end

        # Create output
        new{typeof(built_diagnostics),typeof(u),typeof(u0),typeof(t),typeof(simulation),
            typeof(physical_transform),typeof(h5_kwargs)}(built_diagnostics, u, u0, t,
                                                          simulation, physical_transform,
                                                          store_hdf, store_locally, true,
                                                          h5_kwargs)
    end
end

# --------------------------------- Prepare Initial State ----------------------------------

"""
    prepare_initial_state(prob; physical_transform=identity)

  Prepares the initial state by applying the user defined `physical_transform`, if any, to 
  a copy of the initial condition stored in `prob`, returning it alongside the initial time. 
"""
function prepare_initial_state(prob; physical_transform=identity)
    u0 = physical_transform(copy(prob.u0))
    t0 = first(prob.tspan)
    return u0, t0
end

# -------------------------------------- HDF5 Setup ----------------------------------------

"""
    setup_hdf5_storage(filename, simulation_name, N_samples::Int, u0, prob, t0;
    store_hdf=store_hdf, h5_kwargs=h5_kwargs)

  Creates a *HDF5* file, if not existing, and writes a group with `simulation_name` to it, 
  refered to as a `simulation` group. If the simulation group does not exists, the `h5_kwargs` 
  are applied to the `"fields"` and `"t"` datasets. The opened `simulation` is returned.
"""
function setup_hdf5_storage(filename, simulation_name, u0, prob, t0; store_hdf=store_hdf,
                            h5_kwargs=h5_kwargs)
    if store_hdf
        filename = add_h5_if_missing(filename)

        # Create HDF5 file
        file = h5open(filename, "cw")

        # Create simulation name
        simulation_name = handle_simulation_name(simulation_name, prob)

        # Checks how to handle simulation group
        if !haskey(file, simulation_name)
            # Create simulation group
            simulation = create_group(file, simulation_name)
            # Store attributes
            write_attributes(simulation, prob)
        else
            simulation = open_group(file, simulation_name)
        end

        return simulation
    else
        return nothing
    end
end

# ------------------------------------- HDF5 Helpers ---------------------------------------

"""
    add_h5_if_missing(filename::AbstractString)

  Makes sure the filename has an extension, if not the `.h5` extension is added. 
"""
function add_h5_if_missing(filename::AbstractString)
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
    attributes = setdiff(fieldnames(typeof(domain)),
                         (:x, :y, :kx, :ky, :SC, :transforms, :precision))
    for attribute in attributes
        write_attribute(simulation, string(attribute), getproperty(domain, attribute))
    end
end

# ---------------------------------- Setup Local Storage -----------------------------------

# TODO clean-up here later, make local storage a Dict
"""
    setup_local_storage(u0, t0, N_samples; store_locally=store_locally)
  
  Allocates vectors in memory for storing the fields alongside the time if the user wants it,
  otherwise empty vectors are returned.
"""
function setup_local_storage(u0, t0; store_locally=store_locally)
    if store_locally
        # Allocate local memory for fields
        u = [zero(u0) for _ in 1:N_samples]
        u[1] .= u0
        t = zeros(N_samples)
        t[1] = t0
    else
        u, t = [], []
    end

    return u, t
end

"""
    write_local_state(output, idx, u, t)

  Writes the state `u` at time `t` to the local storage in the `Output` struct.
"""
function write_local_state(output::Output, idx, u, t)
    output.u[idx] .= u
    output.t[idx] = t
end

# ---------------------------- Stride And Storage Size Related -----------------------------

"""
    prepare_sampling(stride::Int, field_storage_limit, prob::SpectralODEProblem)

  Prepares the sampling strategy for storing simulation states based on the desired stride 
  and storage limit. Determines the number of samples to store and the stride between samples.
  
  #### Arguments
  - `stride::Int`: The proposed stride between samples. If set to `-1`, the function 
  will automatically recommend an appropriate stride based on the storage limit.
  - `field_storage_limit`: The maximum allowed storage size for the output of fields, as a 
  string (e.g., `"100 MB"`). The limit does not affect the storage size of the Diagnostics! 
  If empty, no storage constraint is applied.
  - `prob::SpectralODEProblem`: The `SpectralODEProblem`` containing the size of the fields.
    
  #### Notes
  - If both `stride` and `field_storage_limit` are unspecified, all steps are recorded.
  - The function validates and adjusts `stride` to ensure it is within feasible bounds.
  - Issues a warning if the last step has a different stride than the rest.
"""
function prepare_sampling(stride::Int, field_storage_limit, prob::SpectralODEProblem)

    # Compute total number of simulation steps
    N_steps = compute_number_of_steps(prob)

    # Handle all the different scenarios, stride = -1 => let the program decide
    if !isempty(field_storage_limit)
        storage_bytes = parse_storage_limit(field_storage_limit)
        if stride == -1
            stride = recommend_stride(storage_bytes, N_steps, prob)
        else
            check_storage_size(storage_bytes, N_steps, stride, prob)
        end
    elseif stride == -1
        stride = 1
    end

    # Validate stride, might change stride if too large
    stride = validate_stride(N_steps, stride)

    # Compute number of data points to record and warn if last step has differnt step size
    N_samples = cld(N_steps, stride) + 1

    return N_samples, stride
end

"""
    compute_number_of_steps(prob::SpectralODEProblem)
  
  Computes the number of time steps the integrator needs to solve the problem `prob`. The 
  last step might be fractional and is therefore rounded up due to the fixed `prob.dt`.
"""
function compute_number_of_steps(prob::SpectralODEProblem)
    ceil(Int, (last(prob.tspan) - first(prob.tspan)) / prob.dt)
end

"""
    recommend_stride(storage_limit::Int, N_steps::Int, prob::SpectralODEProblem)
  
  Recommends the closest divisor to the minimum stride needed to fullfil the `storage_limit`.
  If the `storage_limit` is too strict an error is thrown, which informs the user of the 
  minimum limit.
"""
function recommend_stride(storage_limit::Int, N_steps::Int, prob::SpectralODEProblem)
    field_bytes = length(prob.u0) * sizeof(eltype(prob.u0))
    # Determine how many fields can be fully stored
    max_samples = storage_limit ÷ field_bytes

    # At least two should be stored, start and end
    if max_samples < 2
        throw(ArgumentError("The storage limit ($(format_bytes(storage_limit))) is too small. \
        The field(s) alone requires $(format_bytes(field_bytes)), however at least two samples \
        are required (minimum limit: $(format_bytes(2*field_bytes)))."))
    end

    # Compute minimum stride to achieve the max number of samples
    min_stride = ceil(Int, N_steps / (max_samples - 1))

    # Picks closest divisor that does not exceed storage limit, while N_steps ≥ min_stride
    recommended = next_divisor(N_steps, min_stride)
    return recommended
end

"""
    check_storage_size(storage_limit::Int, N_steps::Int, stride::Int, prob::SpectralODEProblem)
  
  Checks that the needed storage does not exceed the `storage_limit`, otherwise an error is 
  thrown, which recommends the minimum divisor satisfying the storage limit. In addition the
   error checks of `recommend_stride` are performed, which may trigger before the storage check.
"""
function check_storage_size(storage_limit::Int, N_steps::Int, stride::Int,
                            prob::S) where
         {S<:SpectralODEProblem}
    min_stride = recommend_stride(storage_limit, N_steps, prob)

    storage_need = compute_storage_need(N_steps, stride, prob)
    if storage_need > storage_limit
        throw(ArgumentError("The total output requires $(format_bytes(storage_need)), which exceeds the \
                            storage limit of $(format_bytes(storage_limit)). Consider increasing the \
                            `stride` (minimum recommended: $min_stride) or the `field_storage_limit`."))
    end
end

"""
    compute_storage_need(N_steps::Int, stride::Int, prob::SpectralODEProblem)

  Computes the storage needed to store `N_steps÷stride`samples with `sizeof(prob.u0)`. 
"""
function compute_storage_need(N_steps::Int, stride::Int, prob::SpectralODEProblem)
    stride < 1 ? throw(ArgumentError("stride must be ≥ 1, got $stride")) : nothing
    (cld(N_steps, stride) + 1) * length(prob.u0) * sizeof(eltype(prob.u0))
end

"""
    validate_stride(N_steps::Int, stride::Int)
  
    Validates and adjusts the `stride`. If `stride`:

  - exceeds `N_steps`, it is set to `N_steps` and a warning is issued.
  - is less than 1 an `ArgumentError` is thrown.
  - does not evenly divide `N_steps`, a warning is issued and a suggested divisor is provided.

  Returns the validated (and possibly adjusted) `stride`.
"""
function validate_stride(N_steps::Int, stride::Int)
    if N_steps < stride
        @warn "stride ($stride) exceeds total steps ($N_steps). \
               Adjusting to stride = N_steps ($N_steps)."
        stride = N_steps
    end

    stride < 1 ? throw(ArgumentError("stride must be ≥ 1, got $stride")) : nothing

    if N_steps % stride != 0
        suggestion = nearest_divisor(N_steps, stride)
        @warn "stride ($stride) does not evenly divide N_steps ($N_steps). The \
        final output interval will be shorter. Consider using stride = $suggestion instead."
    end

    return stride
end

"""
    parse_storage_limit(limit::Integer)
    parse_storage_limit(limit::AbstractString)

  Parses a storage limit specified either as an integer (number of bytes) or as a string with 
  optional units (defaults to B). Supports both decimal and binary units up to "EiB", with a 
  decimal number of bytes. For example: `"10MB"`, `"5.2 GiB"`, `"1024"`.
"""
parse_storage_limit(limit::Integer) = limit

function parse_storage_limit(limit::AbstractString)
    # Units dictionary
    units = Dict("B" => 1, "KB" => 10^3, "KIB" => 1024, "MB" => 10^6, "MIB" => 1024^2,
                 "GB" => 10^9, "GIB" => 1024^3, "TB" => 10^12, "TIB" => 1024^4,
                 "PB" => 10^15,
                 "PIB" => 1024^5, "EB" => Int128(10^18), "EIB" => Int128(1024^6))

    # Match numeric value and optional unit
    # \s* matches 0 or more white spaces (\s) at the beginning ^ and end $
    # \d matches digits + -> 1 or more, * 0 or more
    # (?:\.\d) non-capturing group starting with a dot \., ? means matches zero or one
    # [a-zA-Z] matches all combinations of roman letters
    m = match(r"^\s*(\d*(?:\.\d*)?)\s*([a-zA-Z]*)\s*$", limit)
    if m === nothing
        throw(ArgumentError("Invalid storage limit format: $limit"))
    end

    # Get value and unit from Regex
    value = parse(Float64, m.captures[1])
    unit = uppercase(m.captures[2])
    unit = unit == "" ? "B" : unit # Converts no unit to mean byte

    if !haskey(units, unit)
        throw(ArgumentError("Unknown storage unit: $unit"))
    end

    multiplier = units[unit]
    return round(typeof(multiplier), value * multiplier)
end

"""
    format_bytes(nbytes::Integer)
  
  Formats the number of bytes into a "human readable" format. For example: `1032` ⇒ `"1.03 KB"`.
"""
function format_bytes(nbytes::Integer)
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    i = 1
    f = float(nbytes)
    while f >= 1000 && i < length(units)
        f /= 1000
        i += 1
    end
    return string(round(f; digits=2), " ", units[i])
end

"""
    nearest_divisor(N::Int, target::Int)
  Finds the nearest divisor of `N` to the target. If `N` is prime `1` is returned.
"""
function nearest_divisor(N::Int, target::Int)
    closest, smallest = 1, abs(target - 1)
    for d in 1:floor(Int, sqrt(N))
        if N % d == 0
            for c in (d, N ÷ d)
                dist = abs(c - target)
                if dist < smallest || (dist == smallest && c < closest)
                    closest, smallest = c, dist
                end
            end
        end
    end
    return closest
end

"""
    next_divisor(N, target)

  Finds the smallest divisor of `N` that is ≥ `target`. If none exists (i.e. `target > N`), \
  `N` is returned.
"""
function next_divisor(N::Int, target::Int)
    recommended = N
    for d in 1:floor(Int, sqrt(N))
        if N % d == 0
            for c in (d, N ÷ d)
                if c >= target && c < recommended
                    recommended = c
                end
            end
        end
    end
    return recommended
end

# ------------------------------ DIAGNOSTIC BUILDING RELATED -------------------------------

# TODO check if new dim of fields, in that case probably should re-create simulation group
#simulation = setup_simulation_group(file, simulation_name, N_samples, u0, prob,
#                                    t0; h5_kwargs)
"""
    setup_simulation_group(file, simulation_name, N_samples, u0, prob, t0; h5_kwargs)

  Creates a *HDF5* group with `simulation_name` (a "simulation"), and allocates the correct 
  sizes based on `N_samples` with the fields being chunked with additinal `h5_kwargs` applied.
  In addition the inital condition is written along with the attributes of the `prob`.
""" # TODO move logic elsewhere
function setup_simulation_group(file, simulation_name, N_samples, u0, prob, t0; h5_kwargs)

    # Create simulation group
    simulation = create_group(file, simulation_name)

    # Create dataset for fields and time
    dset = create_dataset(simulation, "fields", datatype(eltype(u0)),
                          (size(u0)..., typemax(Int64)); chunk=(size(u0)..., 1),
                          h5_kwargs...)
    HDF5.set_extent_dims(dset, (size(u0)..., N_samples))
    dset = create_dataset(simulation, "t", datatype(Float64), # TODO eltype(t0)) bug if tspan has Int type
                          (typemax(Int64),); chunk=(1,), h5_kwargs...)
    HDF5.set_extent_dims(dset, (N_samples,))

    # Store the initial conditions
    write_state(simulation, 1, u0, t0)

    return simulation
end

"""
    write_state(simulation, idx::Int, u, t)

  Writes the state `u` at time `t` to the simulation group `simulation` (HDF5).
""" # TODO merge with writing of the data
function write_state(simulation, idx::Int, u, t)
    simulation["fields"][fill(:, ndims(u))..., idx] = u
    simulation["t"][idx] = t
end

"""
    reopen_simulation_group!(file, simulation_name, N_samples::Int, u0)

  Reopens an existing `simulation` group and extends its datasets if needed to accommodate 
  a different `size(u0)` and/or `N_samples`.
""" # TODO remove this function and instead add functionality elsewhere
function reopen_simulation_group!(file, simulation_name, N_samples::Int, u0)
    simulation = open_group(file, simulation_name)
    # Expand length of fields and t
    HDF5.set_extent_dims(simulation["fields"], (size(u0)..., N_samples))
    HDF5.set_extent_dims(simulation["t"], (N_samples,))
    return simulation
end

# ---------------------------------- Main Builder Method -----------------------------------

function build_diagnostics(prob, diagnostic, simulation, h5_kwargs, u0, t0;
                           store_hdf=store_hdf, store_locally=store_locally)
    @unpack diagnostic_recipes, tspan, u0_hat = prob
    # TODO make this into a method
    prob_kwargs = (L=L, N=N, u0=u0, u0_hat=u0_hat, domain=domain, tspan=tspan, p=p,
                   operators=operators, dt=dt, remove_modes=remove_modes, kwargs=kwargs)
    t0 = first(tspan)

    diagnostics = Dict{Symbol,Diagnostic}()
    initial_samples = Dict{Symbol,AbstractArray}()
    storage_requirements = Dict{Symbol,Int}()
    # Cumulative counter
    total_storage_requirement = 0

    for recipe in diagnostic_recipes
        name = recipe.name
        # Build diagnostic
        diagnostics[name] = build_diagnostic(Val(Symbol(recipe.method)); prob_kwargs...,
                                             recipe.kwargs...)
        # Sample once and store output
        initial_samples[name] = diagnostics[name](state, prob, t0) # Here need to differentiate between physical and spectral

        # Compute number of samples to be stored and stride distance
        # Compute output size and predict storage need, compare against own storage limit
        N_samples, stride = prepare_sampling(stride, storage_limit, prob)
        # Compute number of samples to be stored and stride distance
        N_samples, stride = prepare_sampling(stride, field_storage_limit, prob)

        storage_shape = (size(sample)..., N_samples)
        storage_requirements[name] = 0 # TODO figure out what

        # Add storage need to a cumulative sum
        total_storage_requirement += storage_requirements[name]
    end

    # Compare cumulative storage need to Output storage need
    output.storage_limit < total_storage_requirement ? error() : nothing

    for i in something
        # Then build diagnostic outputs
        if stores_data
            # Build h5group
            if store_hdf
                # Create Dict entry
            elseif store_locally
            end
        end
    end

    diag = [HasegawaWakatani.build_diagnostic(Val(recipe.name); domain=domain, tspan=tspan,
                                              dt=1e-3,
                                              recipe.kwargs...) for recipe in diagnostics]
end

# function initialize_diagnostic!(diagnostic, simulation, h5_kwargs, u0, prob, t0, 
#                                 store_hdf::Bool=true, store_locally::Bool=true)

#     if diagnostic.stores_data
#         # Calculate number of samples with rounded sampling rate
#         N = floor(Int, N_steps / diagnostic.sample_step) + 1

#         if N_steps % diagnostic.sample_step != 0
#             @warn "($(diagnostic.name)) Note, there is a $(diagnostic.sample_step + N_steps%diagnostic.sample_step) sample step at the end"
#         end
#         if store_hdf
#             if !haskey(simulation, diagnostic.name)
#                 # Create group
#                 diagnostic.h5group = create_group(simulation, diagnostic.name)

#                 # Create dataset for fields and time
#                 ## Datatype and shape is not so trivial here..., will have to think about it tomorrow
#                 dset = create_dataset(simulation[diagnostic.name], "data",
#                                       datatype(eltype(id)), (size(id)..., typemax(Int64));
#                                       chunk=(size(id)..., 1), h5_kwargs...)
#                 HDF5.set_extent_dims(dset, (size(id)..., N))
#                 dset = create_dataset(simulation[diagnostic.name], "t",
#                                       datatype(eltype(id)), (typemax(Int64),);
#                                       chunk=(1,), h5_kwargs...)
#                 HDF5.set_extent_dims(dset, (N,))

#                 # Add labels
#                 create_attribute(diagnostic.h5group, "labels", diagnostic.labels)

#                 # Store initial diagnostic
#                 diagnostic.h5group["data"][fill(:, ndims(id))..., 1] = id
#                 diagnostic.h5group["t"][1] = first(prob.tspan)
#             else
#                 diagnostic.h5group = open_group(simulation, diagnostic.name)
#                 # Extend size of arrays
#                 # Open dataset
#                 dset = open_dataset(diagnostic.h5group, "data")
#                 HDF5.set_extent_dims(dset, (size(id)..., N))
#                 dset = open_dataset(diagnostic.h5group, "t")
#                 HDF5.set_extent_dims(dset, (N,))
#             end
#         end

#         if store_locally
#             # Allocate arrays
#             diagnostic.data = [zero(id) for _ in 1:N] #Vector{typeof(id)}(undef, N)
#             diagnostic.t = zeros(N)

#             # Store intial diagnostic
#             if isa(id, AbstractArray)
#                 diagnostic.data[1] .= id
#             else
#                 diagnostic.data[1] = copy(id)
#             end
#             diagnostic.t[1] = first(prob.tspan)
#         end
#     end
# end

# function apply_diagnostic!(diagnostic::D, step::Integer, u::U, prob::SOP, t::N;
#                              store_hdf::Bool=true,
#                              store_locally::Bool=true) where {D<:Diagnostic,
#                                                               U<:AbstractArray,
#                                                               SOP<:SpectralODEProblem,
#                                                               N<:Number}
#     # u might be real or complex depending on previous handle_output and diagnostic.assumes_spectral_state

#     data = diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)

#     if !isnothing(data)
#         # Calculate index
#         idx = step ÷ diagnostic.sample_step + 1

#         store_hdf ? write_data(diagnostic, idx, data, t) : nothing

#         store_locally ? write_local_data(diagnostic, idx, data, t) : nothing
#     end
# end

# TODO REMOVE OR REWRITE THE THREE METHODS BELOW

# TODO perhaps make more like write_state
function write_data(diagnostic, idx, data, t)
    # TODO better check on ndims
    diagnostic.h5group["data"][fill(:, ndims(data))..., idx] = data
    diagnostic.h5group["t"][idx] = t
end

# TODO perhaps same name as write_local_state, different dispatch
function write_local_data(diagnostic::Diagnostic, idx, data, t)
    if isa(data, AbstractArray)
        diagnostic.data[idx] .= data
    else
        diagnostic.data[idx] = copy(data)
    end
    diagnostic.t[idx] = t
end

function apply_diagnostic(diagnostic, state, prob, time)
    # Take diagnostic of initial field (id = initial diagnostic)
    #if diagnostic.assumes_spectral_state
    #  id = diagnostic(prob.u0_hat, prob, first(prob.tspan))
    #else
    #  id = diagnostic(u0, prob, first(prob.tspan))
    #end
end

"""
  # Initialization
  prepare_sampling

  # Storage size magic 
  compute_number_of_steps
  recommend_stride
  check_storage_size
  compute_storage_need
  validate_stride
  parse_storage_limit
  format_bytes
  nearest_divisor
  next_divisor
    
  # In memory storage
  setup_local_storage

  # HDF5
  create_or_open_group 
  reopen_simulation_group
  rewrite_dataset
  setup_simulation_group

  # Sampling
  sample_diagnostic!
  transform_state!

  # Writing
  write_state
  write_local_state
"""

# ---------------------------------- Handling Of Output ------------------------------------

"""
    handle_output!(output::O, step::Integer, u::T, prob::SOP, t::N)

  Handles output operations for a simulation step, including state storage, diagnostics 
  sampling, and mode removal. Ensures the spectral state is only transformed once per step. 
  In addition a check is performed to detect breakdowns, to throw an error. 
"""
function handle_output!(output::O, step::Integer, state::T, prob::SOP,
                        time::N) where {O<:Output,T<:AbstractArray,SOP<:
                                        SpectralODEProblem,N<:Number}

    # Keeps track such that state only transformed once
    output.transformed = false

    # Remove modes after each step using user defined function
    prob.remove_modes(state, prob.domain)

    # Handle diagnostics
    maybe_sample_diagnostics!(output, step, state, prob, time)

    # TODO make it time based
    if step % 1000 == 0
        output.store_hdf ? flush(output.simulation.file) : nothing
    end

    # Check if first value is NaN, if one value is NaN the whole Array will turn NaN after FFT
    assert_no_nan(state, time)
end

"""
    maybe_sample_diagnostics!(output, step::Integer, u, prob, t)

  Iterates through the list of diagnostics (`output.diagnostics`) and determines whether or
  not to sample the diagnostic.
"""
function maybe_sample_diagnostics!(output, step::Integer, state, prob, time)
    # Handle diagnostics
    for diagnostic in output.diagnostics
        if step % diagnostic.stride == 0
            sample_diagnostic!(output, diagnostic, step, state, prob, time)
        end
    end
end

"""
   TODO write actuall string: The spectral state `u` is transformed to the real
  state `U`, with the user defined `physical_transform` applied, before being stored.
"""
function sample_diagnostic!(output, diagnostic, step::Integer, u, prob, t)
    # Check if diagnostic assumes physical field and transform if not yet done
    if !diagnostic.assumes_spectral_state && !output.transformed
        # Transform state
        transform_state!(output, u, get_bwd(prob.domain))
    end

    # Passes the logic onto perform_diagnostic! to do diagnostic and store data
    if diagnostic.assumes_spectral_state
        perform_diagnostic!(diagnostic, step, u, prob, t;
                            store_hdf=output.store_hdf, store_locally=output.store_locally)
    else
        perform_diagnostic!(diagnostic, step, output.U_buffer, prob, t;
                            store_hdf=output.store_hdf, store_locally=output.store_locally)
    end

    # Store state
    #store_state(output, step, output.U_buffer, t)
end

"""
    transform_state!(output, u, p)

  Transforms spectral coefficients `u_hat` into real fields by applying `spectral_transform!` 
  to the `output.U_buffer` buffer. The user defined `physical_transform` is also applied to 
  the buffer, and the `output.transformed` flag is updated to not transform same field twice.
"""
function transform_state!(output, u_hat, p)
    spectral_transform!(output.U_buffer, p, u_hat)
    output.physical_transform(output.U_buffer)
    output.transformed = true
end

# TODO remove
"""
    store_state(output, step::Integer, U, t)

  Stores state to *HDF5* file and memory, depending on the state of the `output.store_hdf` 
  and `output.store_locally` respectively. The index is computed based on the step.
"""
function store_state(output, step::Integer, U, t)
    # Calculate index
    idx = step ÷ output.stride + 1

    # Store in hdf if user wants
    output.store_hdf ? write_state(output.simulation, idx, U, t) : nothing

    # Store in memory if user wants
    output.store_locally ? write_local_state(output, idx, U, t) : nothing
end

"""
    # Store final state
    # function write_final_state()
    # Get finale index
    #idx = "end"

    #     # Store in hdf if user wants
    #     output.store_hdf ? write_state(output.simulation, idx, U, t) : nothing

    #     # Store in locally if user wants
    #     output.store_locally ? write_local_state(output, idx, U, t) : nothing
    # end
"""

# -------------------------------------- Checkpoint ----------------------------------------

"""
    save_checkpoint!(output::O, cache::C, step::Integer, t::N) where {O<:Output,
    C<:AbstractCache,N<:Number}

  Creates or opens a `checkpoint` and stors the `cache`at time `t` corresponding to step=`step`.
"""
function save_checkpoint!(output::O, cache::C, step::Integer,
                          time::N) where {O<:Output,
                                          C<:AbstractCache,N<:Number}
    if output.store_hdf
        # Create or open a h5group for the checkpoint
        checkpoint = create_or_open_group(output.simulation, "checkpoint")
        # Store checkpoint
        store_checkpoint!(checkpoint, cache, step, time)
    end
end

"""
    store_checkpoint!(checkpoint::G, cache::C, step::Integer, t::N) where {G<:HDF5.Group,
    C<:AbstractCache,N<:Number}

  Stores checkpoint by dumping the fields of the `cache` that are not of type 
  `AbstractTableau`. This overwrites the previously stored checkpoint.
""" # Perhaps one could look into HDF5 compound types in the future
function store_checkpoint!(checkpoint::G, cache::C, step::Integer,
                           time::N) where {G<:HDF5.Group,
                                           C<:AbstractCache,N<:Number}

    # Get all the attributes of cache
    keys = fieldnames(typeof(cache))

    # Dump cache
    for key in keys
        val = getfield(cache, key)
        # Do not want to bother with storing Tableau, easy to recover
        isa(val, AbstractTableau) ? continue : nothing

        # Convert to Array for easy handling, also downloads from GPU in case of CPUArray
        isa(val, AbstractArray) ? val = Array(val) : nothing

        # Backup the field
        rewrite_dataset(checkpoint, string(key), val)
    end

    # Backup step and time
    rewrite_dataset(checkpoint, "time", time)
    rewrite_dataset(checkpoint, "step", step)
end

"""
    restore_checkpoint(simulation::HDF5.Group, prob::SOP, scheme::SA) where {
    SOP<:SpectralODEProblem,SA<:AbstractODEAlgorithm}

  Restores Cache for `scheme` from checkpoint stored in `simulation`.
"""
function restore_checkpoint(simulation::HDF5.Group, prob::SOP,
                            scheme::SA) where {
                                               SOP<:SpectralODEProblem,
                                               SA<:AbstractODEAlgorithm}

    #validate_simulation_group TODO check that dt remains the same, and other parameters

    # Create cache container and get the fieldnames
    cache = get_cache(prob, scheme)
    keys = fieldnames(typeof(cache))

    # Restore cache
    for key in keys
        field = getfield(cache, key)
        # Skip restoring Tableau
        isa(field, AbstractTableau) ? continue : nothing

        # Adapt the data to the same type as the field
        data = adapt(typeof(field), read(simulation["checkpoint"], string(key)))
        setproperty!(cache, key, data)
    end

    return cache
end

# --------------------------------------- Utilities ----------------------------------------

# TODO detangle this mess
"""
    parameter_string(parameters::AbstractDict)
    parameter_string(parameters::AbstractDict)

  Creates a human readable string listing the parameters with values, for example: 
  `(κ=0.1,σ=0.02)` ⇒ `"κ=0.1, σ=0.02"`.
"""
function parameter_string(parameters::P) where {P<:AbstractDict}
    tmp = [string(key, "=", value) for (key, value) in sort(collect(parameters))]
    join(tmp, ", ")
end

function parameter_string(parameters::P) where {P<:NamedTuple}
    tmp = [string(key, "=", value) for (key, value) in sort(collect(pairs(parameters)))]
    join(tmp, ", ")
end

"""
    assert_no_nan(u::AbstractArray, t)
    assert_no_nan(u::AbstractGPUArray, t)

  Checks if the first entry in `u` is `NaN`, if so a breakdown occured and an error is thrown.
"""
assert_no_nan(state::AbstractArray, time) =
    if isnan(state[1])
        error("Breakdown occured at t=$time")
    end

function assert_no_nan(state::AbstractGPUArray, time)
    @allowscalar isnan(state[1]) ? error("Breakdown occured at t=$time") : nothing
end

"""
    create_or_open_group(parent::Union{HDF5.File,HDF5.Group}, path::AbstractString; properties...)

  Creates or opens an HDF5 group at the specified `path` within the given `parent` (which can 
  be an `HDF5.File` or `HDF5.Group`). If the group at `path` exists, it is opened using 
  `open_group`; otherwise, a new group is created using `create_group`. Additional keyword 
  arguments (`properties...`) are passed onto the group creation or opening functions.
"""
function create_or_open_group(parent::Union{HDF5.File,HDF5.Group}, path::AbstractString;
                              properties...)
    if haskey(parent, path)
        return open_group(parent, "checkpoint", properties...)
    else
        return create_group(parent, "checkpoint", properties...)
    end
end

"""
    rewrite_dataset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, data; pv...)

  Deletes the dataset if it allready exists and creates and writes to a new dataset.
"""
function rewrite_dataset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, data;
                         pv...)
    if haskey(parent, name)
        delete_object(parent[name])
    end
    write_dataset(parent, name, data; pv...)
end

# ---------------------------------------- Helpers -----------------------------------------

function Base.show(io::IO, m::MIME"text/plain", output::Output)
    print(io, "Output (store_hdf=", output.store_hdf,
          ", store_locally=", output.store_locally, ")")
    output.store_hdf ?
    print(io, ", simulation: ", HDF5.name(output.simulation), " (file: ",
          output.simulation.file.filename, "):") : print(":")

    if output.physical_transform !== identity
        print(io, "\nphysical_transform: ", nameof(output.physical_transform))
    end

    print(io, "\nDiagnostics:")
    for diagnostic in output.diagnostics
        print(io, "\n-")
        show(io, m, diagnostic)
    end
end

import Base.close
close(output::Output) = close(output.simulation.file)