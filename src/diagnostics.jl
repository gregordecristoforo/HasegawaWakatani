using LinearAlgebra
using Plots
using FFTW
using Printf
using Interpolations
import PlotlyJS

# ----------------------------------- Diagnostics ------------------------------------------

mutable struct Diagnostic{N<:AbstractString,M<:Function,D<:AbstractArray,T<:AbstractArray,
    HG<:Union{Nothing,HDF5.Group},L<:Any,A<:Tuple,K<:NamedTuple}

    name::N
    method::M
    data::D
    t::T
    sampleStep::Int
    h5group::HG
    label::L
    assumesSpectralField::Bool
    storesData::Bool
    args::A
    kwargs::K

    function Diagnostic(name::N, method::M, sampleStep::Int=-1, label::L="", args::A=(),
        kwargs::K=NamedTuple(); assumesSpectralField::Bool=false, storesData::Bool=true) where {
        N<:AbstractString,M<:Function,L<:Any,A<:Tuple,K<:NamedTuple}
        new{typeof(name),typeof(method),Vector,Vector,Union{Nothing,HDF5.Group},typeof(label),
            typeof(args),typeof(kwargs)}(name, method, Vector[], Vector[], sampleStep, nothing,
            label, assumesSpectralField, storesData, args, kwargs)
    end
end

function initialize_diagnostic!(diagnostic::D, U::T, prob::SOP, simulation::S, h5_kwargs::K;
    store_hdf::Bool=true, store_locally::Bool=true) where {D<:Diagnostic,T<:AbstractArray,
    SOP<:SpectralODEProblem,S<:Union{HDF5.Group,Nothing},K<:Any}

    # Calculate total number of steps
    N_steps = floor(Int, (last(prob.tspan) - first(prob.tspan)) / prob.dt)

    # If user did not specify 
    if diagnostic.sampleStep == -1
        #TODO implement some logic here later
        diagnostic.sampleStep = 1
        # N_data = floor(Int, 0.1 * N_steps)

        # if N_data > N_steps
        #     N_data = N_steps + 1
        #     @warn "N_data and stepsize was not compatible, N_data is instead set to N_data = " * "$N_data"
        # end

        # # Calculate number of evolution steps between samples
        # fieldStep = floor(Int, N_steps / (N_data - 1))
    end

    if diagnostic.sampleStep > N_steps
        diagnostic.sampleStep = N_steps
        @warn "The sample step was larger than the number of steps and has been set to sampleStep = $N_steps"
    end

    # Take diagnostic of initial field (id = initial diagnostic)
    if diagnostic.assumesSpectralField
        id = diagnostic.method(prob.u0_hat, prob, first(prob.tspan), diagnostic.args...; diagnostic.kwargs...)
    else
        id = diagnostic.method(U, prob, first(prob.tspan), diagnostic.args...; diagnostic.kwargs...)
    end

    if diagnostic.storesData
        # Calculate number of samples with rounded sampling rate
        N = floor(Int, N_steps / diagnostic.sampleStep) + 1

        if N_steps % diagnostic.sampleStep != 0
            @warn "($(diagnostic.name)) Note, there is a $(diagnostic.sampleStep + N_steps%diagnostic.sampleStep) sample step at the end"
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

                # Add label
                create_attribute(diagnostic.h5group, "label", diagnostic.label)

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

function perform_diagnostic!(diagnostic::D, step::Int, u::U, prob::SOP, t::N;
    store_hdf::Bool=true, store_locally::Bool=true) where {D<:Diagnostic,U<:AbstractArray,
    SOP<:SpectralODEProblem,N<:Number}
    # u might be real or complex depending on previous handle_output and diagnostic.assumesSpectralField

    if diagnostic.storesData
        # Calculate index
        idx = step ÷ diagnostic.sampleStep + 1

        # Perform diagnostic
        data = diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)

        if store_hdf
            # TODO better check on ndims
            diagnostic.h5group["data"][fill(:, ndims(data))..., idx] = data
            diagnostic.h5group["t"][idx] = t
        end

        if store_locally
            if isa(data, AbstractArray)
                diagnostic.data[idx] .= data
            else
                diagnostic.data[idx] = copy(data)
            end
            diagnostic.t[idx] = t
        end
    else
        # Apply diagnostic
        diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)
    end
end

#-------------------------------------- Include --------------------------------------------

include("diagnostics/probe.jl")
include("diagnostics/COM.jl")
include("diagnostics/CFL.jl")
include("diagnostics/display.jl")
include("diagnostics/spectral.jl")
include("diagnostics/boundary.jl")
include("diagnostics/projection.jl")
include("diagnostics/parameter_study.jl")
include("diagnostics/progress.jl")
include("diagnostics/profiles.jl")
include("diagnostics/energy_integrals.jl")
include("diagnostics/fluxes.jl")

# Default diagnostic
#cflDiagnostic = Diagnostic(CFLExB, 100, "cfl")
const DEFAULT_DIAGNOSTICS = [ProgressDiagnostic()]

# ---------------------------------- Other -------------------------------------------------

# Enstrophy dissipation