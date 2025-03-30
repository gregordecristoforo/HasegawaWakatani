using LinearAlgebra
using Plots
using FFTW
using Printf
using Interpolations
import PlotlyJS

# ----------------------------------- Diagnostics ------------------------------------------

# TODO probabily should split diagnostic up to smaller more maintainable files
# TODO evaluate wether or not to be mutable if push!(data) is used does not need to be mutable
mutable struct Diagnostic
    name::AbstractString
    method::Function
    data::AbstractArray
    t::AbstractArray
    sampleStep::Integer
    h5group::Any #::HDF5.Group
    label::Any
    assumesSpectralField::Bool
    storesData::Bool
    args::Tuple
    kwargs::NamedTuple

    function Diagnostic(name::String, method::Function, sampleStep::Integer=-1, label="", args=(), kwargs=NamedTuple(); assumesSpectralField=false, storesData=true)
        new(name, method, Vector[], Vector[], sampleStep, nothing, label, assumesSpectralField, storesData, args, kwargs)
    end
end

function initialize_diagnostic!(diagnostic::Diagnostic, prob, simulation, h5_kwargs) #::SpectralODEProblem

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
        U = copy(prob.u0)
        #prob.recover_fields!(U)
        id = diagnostic.method(U, prob, first(prob.tspan), diagnostic.args...; diagnostic.kwargs...)
    end

    if diagnostic.storesData
        # Calculate number of samples with rounded sampling rate
        N = floor(Int, N_steps / diagnostic.sampleStep) + 1

        if N_steps % diagnostic.sampleStep != 0
            @warn "($(diagnostic.name)) Note, there is a $(diagnostic.sampleStep + N_steps%diagnostic.sampleStep) 
                    sample step at the end"
        end

        # Create group
        diagnostic.h5group = create_group(simulation, diagnostic.name)

        # Create dataset for fields and time
        ## Datatype and shape is not so trivial here..., will have to think about it tomorrow
        dset = create_dataset(simulation[diagnostic.name], "data", datatype(Float64), (size(id)..., typemax(Int64)),
            chunk=(size(id)..., 1); h5_kwargs...)
        HDF5.set_extent_dims(dset, (size(id)..., N))
        dset = create_dataset(simulation[diagnostic.name], "t", datatype(Float64), (typemax(Int64),),
            chunk=(1,); h5_kwargs...)
        HDF5.set_extent_dims(dset, (N,))

        # Add label
        create_attribute(diagnostic.h5group, "label", diagnostic.label)

        # Allocate arrays
        diagnostic.data = Vector{typeof(id)}(undef, N)
        diagnostic.data[1] = id
        diagnostic.h5group["data"][fill(:, ndims(id))..., 1] = id
        diagnostic.t = zeros(N)
        diagnostic.t[1] = first(prob.tspan)
        diagnostic.h5group["t"][1] = first(prob.tspan)
    end
end

function perform_diagnostic!(diagnostic::Diagnostic, step::Integer, u::AbstractArray, prob::SpectralODEProblem, t::Number)
    if diagnostic.storesData
        idx = step ÷ diagnostic.sampleStep + 1

        if diagnostic.assumesSpectralField
            diagnostic.data[idx] = diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)
        else
            U = real(spectral_transform(u, prob.domain.transform.iFT)) # transform to realspace
            #prob.recover_fields!(U) # apply a transformation, for instance exp to get n from log(n)
            diagnostic.data[idx] = diagnostic.method(U, prob, t, diagnostic.args...; diagnostic.kwargs...)
        end
        # TODO better check on ndims
        diagnostic.h5group["data"][fill(:, ndims(diagnostic.data[idx]))..., idx] = diagnostic.data[idx]
        diagnostic.t[idx] = t
        diagnostic.h5group["t"][idx] = t
    else
        if diagnostic.assumesSpectralField
            diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)
        else
            U = real(spectral_transform(u, prob.domain.transform.iFT)) # transform to realspace
            #prob.recover_fields!(U)
            diagnostic.method(U, prob, t, diagnostic.args...; diagnostic.kwargs...)
        end
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
include("diagnostics/energy_integrals.jl")

# ---------------------------------- Other -------------------------------------------------

# Total energy

# Density energy
# function P(u)
#     0.5*u[:,:,1]
# end

# # Kinetic energy
# function K(u)
#     0.5*u[:,:,1]
# end

# Enstrophy

# Density flux

# Resistive flux

# Energy dissipation

# Enstrophy dissipation
