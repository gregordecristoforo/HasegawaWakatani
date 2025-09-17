# ----------------------------------- Diagnostics ------------------------------------------

mutable struct Diagnostic{N<:AbstractString,M<:Function,D<:AbstractArray,T<:AbstractArray,
    HG<:Union{Nothing,HDF5.Group},L<:Any,A<:Tuple,K<:NamedTuple}

    name::N
    method::M
    data::D
    t::T
    sample_step::Int
    h5group::HG
    labels::L
    assumes_spectral_field::Bool
    stores_data::Bool
    args::A
    kwargs::K

    function Diagnostic(name::N, method::M, sample_step::Int=-1, labels::L="", args::A=(),
        kwargs::K=NamedTuple(); assumes_spectral_field::Bool=false, stores_data::Bool=true) where {
        N<:AbstractString,M<:Function,L<:Any,A<:Tuple,K<:NamedTuple}
        new{typeof(name),typeof(method),Vector,Vector,Union{Nothing,HDF5.Group},typeof(labels),
            typeof(args),typeof(kwargs)}(name, method, Vector[], Vector[], sample_step, nothing,
            labels, assumes_spectral_field, stores_data, args, kwargs)
    end
end

function initialize_diagnostic!(diagnostic::D, prob::SOP, u0::T, t0::AbstractFloat,
    simulation::S, h5_kwargs::K; store_hdf::Bool=true, store_locally::Bool=true) where
{D<:Diagnostic,SOP<:SpectralODEProblem,T<:AbstractArray,S<:Union{HDF5.Group,Nothing},K<:Any}

    # Calculate total number of steps
    N_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / prob.dt)

    # If user did not specify 
    if diagnostic.sample_step == -1
        #TODO implement some logic here later
        diagnostic.sample_step = 1
        # N_data = floor(Int, 0.1 * N_steps)

        # if N_data > N_steps
        #     N_data = N_steps + 1
        #     @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        # end

        # # Calculate number of evolution steps between samples
        # fieldStep = floor(Int, N_steps / (N_data - 1))
    end

    if diagnostic.sample_step > N_steps
        diagnostic.sample_step = N_steps
        @warn "The sample step was larger than the number of steps and has been set to sample_step = $N_steps"
    end

    # Take diagnostic of initial field (id = initial diagnostic)
    if diagnostic.assumes_spectral_field
        id = diagnostic.method(prob.u0_hat, prob, first(prob.tspan), diagnostic.args...; diagnostic.kwargs...)
    else
        id = diagnostic.method(u0, prob, first(prob.tspan), diagnostic.args...; diagnostic.kwargs...)
    end

    if diagnostic.stores_data
        # Calculate number of samples with rounded sampling rate
        N = floor(Int, N_steps / diagnostic.sample_step) + 1

        if N_steps % diagnostic.sample_step != 0
            @warn "($(diagnostic.name)) Note, there is a $(diagnostic.sample_step + N_steps%diagnostic.sample_step) sample step at the end"
        end
        if store_hdf
            if !haskey(simulation, diagnostic.name)
                # Create group
                diagnostic.h5group = create_group(simulation, diagnostic.name)

                # Create dataset for fields and time
                ## Datatype and shape is not so trivial here..., will have to think about it tomorrow
                dset = create_dataset(simulation[diagnostic.name], "data", datatype(eltype(id)), (size(id)..., typemax(Int64)),
                    chunk=(size(id)..., 1); h5_kwargs...)
                HDF5.set_extent_dims(dset, (size(id)..., N))
                dset = create_dataset(simulation[diagnostic.name], "t", datatype(eltype(id)), (typemax(Int64),),
                    chunk=(1,); h5_kwargs...)
                HDF5.set_extent_dims(dset, (N,))

                # Add labels
                create_attribute(diagnostic.h5group, "labels", diagnostic.labels)

                # Store initial diagnostic
                diagnostic.h5group["data"][fill(:, ndims(id))..., 1] = id
                diagnostic.h5group["t"][1] = first(prob.tspan)
            else
                diagnostic.h5group = open_group(simulation, diagnostic.name)
                # Extend size of arrays
                # Open dataset
                dset = open_dataset(diagnostic.h5group, "data")
                HDF5.set_extent_dims(dset, (size(id)..., N))
                dset = open_dataset(diagnostic.h5group, "t")
                HDF5.set_extent_dims(dset, (N,))
            end
        end

        if store_locally
            # Allocate arrays
            diagnostic.data = [zero(id) for _ in 1:N] #Vector{typeof(id)}(undef, N)
            diagnostic.t = zeros(N)

            # Store intial diagnostic
            if isa(id, AbstractArray)
                diagnostic.data[1] .= id
            else
                diagnostic.data[1] = copy(id)
            end
            diagnostic.t[1] = first(prob.tspan)
        end
    end
end

function perform_diagnostic!(diagnostic::D, step::Integer, u::U, prob::SOP, t::N;
    store_hdf::Bool=true, store_locally::Bool=true) where {D<:Diagnostic,U<:AbstractArray,
    SOP<:SpectralODEProblem,N<:Number}
    # u might be real or complex depending on previous handle_output and diagnostic.assumes_spectral_field

    # Perform diagnostic
    data = diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)

    if !isnothing(data)
        # Calculate index
        idx = step ÷ diagnostic.sample_step + 1

        store_hdf ? write_data(diagnostic, idx, data, t) : nothing

        store_locally ? write_local_data(diagnostic, idx, data, t) : nothing
    end
end

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

function Base.show(io::IO, m::MIME"text/plain", diagnostic::Diagnostic)
    print(io, diagnostic.name, " (stride: ", diagnostic.sample_step, ", spectral=",
        diagnostic.assumes_spectral_field, ", stores_data=", diagnostic.stores_data, ")")
    length(diagnostic.args) != 0 ? print(io, ", args=", diagnostic.args) : nothing
    length(diagnostic.kwargs) != 0 ? print(io, ", kwargs=", diagnostic.kwargs) : nothing
end

#-------------------------------------- Include --------------------------------------------

include("probe.jl")
include("COM.jl")
include("CFL.jl")
include("display.jl")
include("spectral.jl")
include("boundary.jl")
include("projection.jl")
include("parameter_study.jl")
include("progress.jl")
include("profiles.jl")
include("energy_integrals.jl")
include("fluxes.jl")

# Default diagnostic
#cflDiagnostic = Diagnostic(CFLExB, 100, "cfl")
const DEFAULT_DIAGNOSTICS = [ProgressDiagnostic()]

# ---------------------------------- Other -------------------------------------------------

# Enstrophy dissipation