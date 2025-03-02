using LinearAlgebra
using Plots
using FFTW
using Printf
using Interpolations
import PlotlyJS

# Extend plotting to allow domain as input
import Plots.plot
function plot(domain::Domain, args...; kwargs...)
    plot(domain.x, domain.y, args...; kwargs...)
end

# Extending PlotlyJS to easily plot surfaces when using Plots for academic figures
function plotlyjsSurface(args...; kwargs...)
    i = findfirst(k -> k === :z, keys(kwargs))
    kwargs = collect(pairs(kwargs))
    kwargs[i] = :z => transpose(kwargs[i][2])
    PlotlyJS.plot(PlotlyJS.surface(args...; kwargs...))
end

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
        prob.recover_fields!(U)
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
            U = real(transform(u, prob.domain.transform.iFT)) # transform to realspace
            prob.recover_fields!(U) # apply a transformation, for instance exp to get n from log(n)
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
            U = real(transform(u, prob.domain.transform.iFT)) # transform to realspace
            prob.recover_fields!(U)
            diagnostic.method(U, prob, t, diagnostic.args...; diagnostic.kwargs...)
        end
    end
end

# --------------------------------- Probe --------------------------------------------------

# probe/high time resolution (fields, velocity etc...)

function probe_field(u::AbstractArray, domain::Domain, positions; interpolation=nothing)
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    # Initilize vectors
    data = zeros(length(positions))

    if isnothing(interpolation)
        for n in eachindex(positions)
            i = argmin(abs.(domain.x .- positions[n][1]))
            j = argmin(abs.(domain.y .- positions[n][2]))
            data[n] = u[j, i]
        end
    else
        # Only want to do this once
        U = interpolation((domain.y, domain.x), u)
        for n in eachindex(positions)
            data[n] = U(positions[n][2], positions[n][1])
        end
    end

    return data
end

function probe_density(u::AbstractArray, prob::SpectralODEProblem, t::Number, positions; interpolation=nothing)
    probe_field(u[:, :, 1], prob.domain, positions; interpolation)
end

# "Constructor" for density probe
function ProbeDensityDiagnostic(positions; interpolation=nothing, N=100)::Diagnostic
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Density probe", probe_density, N, label, args, kwargs)
end

function probe_vorticity(u::AbstractArray, prob::SpectralODEProblem, t::Number, positions; interpolation)
    probe_field(u[:, :, 2], prob.domain, positions; interpolation)
end

# "Constructor" for vorticity probe
function ProbeVorticityDiagnostic(positions; interpolation=nothing, N=100)::Diagnostic
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Vorticity probe", probe_vorticity, N, label, args, kwargs)
end

# TODO implement remaining probes 

# function ProbePhiDiagnostic(x::Union{AbstractArray,Number}, y::Union{AbstractArray,Number};
#     interpolation=nothing, N=100)::Diagnostic
#     args = (x, y)
#     kwargs = (interpolation = interpolation)
#     return Diagnostic(probe_field, N, "probe", args, kwargs)
# end

# function ProbeVelocityDiagnostic(x::Union{AbstractArray,Number}, y::Union{AbstractArray,Number};
#     interpolation=nothing, N=100)::Diagnostic
#     args = (x, y)
#     kwargs = (interpolation = interpolation)
#     return Diagnostic(probe_field, N, "probe", args, kwargs)
# end

# ------------------------------------ COM -------------------------------------------------

# TODO add argument for using quadratures
function radial_COM(u, prob, t, p)
    # 2:end is because the boundaries are periodic and thus should not contribute
    X_COM = sum(prob.domain.x[2:end]' .* u[2:end, 2:end, 1]) / sum(u[2:end, 2:end, 1])

    # Check that do not divide by zero
    if p["previous_time"] == t
        V_COM = 0
    else
        V_COM = (X_COM .- p["previous_position"]) ./ (t .- p["previous_time"])
    end

    # Store 
    p["previous_position"] = X_COM
    p["previous_time"] = t
    [X_COM, V_COM]
end

# "Constructor"
function RadialCOMDiagnostic(N=100)
    #kwargs = (previous_position=0, previous_time=0)
    args = (Dict("previous_position" => 0.0, "previous_time" => 0.0),)
    return Diagnostic("RadialCOMDiagnostic", radial_COM, N, "X_COM, V_COM", args)#, kwargs)
end

#---------------------------------- CFL ----------------------------------------------------

# Calculate velocity assuming U_ExB = ̂z×∇Φ   
function vExB(u::AbstractArray, domain::Domain)
    W = u[:, :, 2] #Assumption
    W_hat = domain.transform.FT * W
    phi_hat = solvePhi(W_hat, domain)
    domain.transform.iFT * -diffY(phi_hat, domain), domain.transform.iFT * diffX(phi_hat, domain)
end

#contourf(vExB([u0;;;u0], domain)[1].^2 .+ vExB([u0;;;u0], domain)[2].^2) 

#Returns max CFL
function cfl_ExB(u::AbstractArray, prob::SpectralODEProblem, t::Number)
    v_x, v_y = vExB(u, prob.domain)
    #(CFLx, CFLy, x, y)
    # if maximum(v_x) * prob.dt / prob.domain.dx >= 0.5
    #     println("Breakdown t=$t")
    # elseif maximum(v_y) * prob.dt / prob.domain.dx >= 0.5
    #     println("Breakdown t=$t")
    # end
    println("CFL", maximum(v_x) * prob.dt / prob.domain.dx)
    [maximum(v_x) * prob.dt / prob.domain.dx, maximum(v_y) * prob.dt / prob.domain.dy]
end

function CFLDiagnostic(N=100)
    Diagnostic("ExB cfl", cfl_ExB, N, "max CFL x, max CFL y")
end

function radial_cfl_ExB(u::AbstractArray, prob::SpectralODEProblem, t::Number, v::Function=vExB)
    v_x, v_y = v(u, prob.domain)
    [maximum(v_x) * prob.dt / domain.dx, argmax(v_x)]
end

function RadialCFLDiagnostic(N=100)
    Diagnostic("Radial CFL", radial_cfl_ExB, N, "max radial CFL, position")
end

function cfl_y(u::AbstractArray, prob::SpectralODEProblem, t::Number, v::Function=vExB)
    v_x, v_y = v(u, prob.domain)
    [maximum(v_y) * prob.dt / domain.dy, argmax(v_y)]
end

# TODO test Burger and maybe remove it and include in the above functions
# CFL where field is velocity
function burgerCFL(u::AbstractArray, prob::SpectralODEProblem, t::Number)
    maximum(u) * prob.dt / prob.domain.dy
end

function BurgerCFLDiagnostic(N=100)
    Diagnostic("Burger CFL", burgerCFL, N, "CFL")
end

#---------------------------- Display diagnostic -------------------------------------------

function plot_density(u::AbstractArray, prob, t::Number)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, u[:, :, 1], aspect_ratio=:equal, xlabel="x", ylabel="y",
        title="n(x, t = $(round(t, digits=digits)))"))
end

function PlotDensityDiagnostic(N=1000)
    Diagnostic("Plot density", plot_density, N, "Display density", storesData=false)
end

function plot_vorticity(u::AbstractArray, prob, t::Number)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, u[:, :, 2], aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Omega" * "(x, t = $(round(t, digits=digits)))", color=:jet))
end

function PlotVorticityDiagnostic(N=1000)
    Diagnostic("Plot vorticity", plot_vorticity, N, "Display vorticity", storesData=false)
end

function plot_potential(u::AbstractArray, prob, t::Number)
    d = prob.domain
    phi = d.transform.iFT * solvePhi(u[:, :, 2], d)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, phi, aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Phi" * "(x, t = $(round(t, digits=digits)))"))
end

function PlotPotentialDiagnostic(N=1000)
    Diagnostic("Plot potential", plot_potential, N, "Display potential", assumesSpectralField=true, storesData=false)
end

#-------------------------------------- Spectral -------------------------------------------

function get_modes(u::AbstractArray, prob, t::Number)
    return u
end

function GetModeDiagnostic(N=100)
    Diagnostic("Mode diagnstic", get_modes, N, "Display density", assumesSpectralField=true)
end

function get_log_modes(u::AbstractArray, prob, t::Number; kx=:ky)
    if kx == :ky
        if length(size(u)) >= 3
            data = zeros(prob.domain.Nx ÷ 2, last(size(u)))
        else
            data = zeros(prob.domain.Nx ÷ 2)
        end

        for i in 1:prob.domain.Nx÷2
            data[i, :] = log.(abs.(u[i, i, :]))
        end
        return data
    else
        return log.(abs.(u[:, kx, :]))
    end
end

function GetLogModeDiagnostic(N=100, kx=:ky)
    Diagnostic("Log mode diagnstic", get_log_modes, N, "log(|u_k|)", assumesSpectralField=true, (), (kx=kx,))
end

function plotFrequencies(u)
    heatmap(log10.(norm.(u)), title="Frequencies")
end

# ------------------------------------------- Boundary diagnostics ---------------------------------------------------------
function lowerXBoundary(u::Array)
    u[1, :]
end

function upperXBoundary(u::Array)
    u[end, :]
end

function lowerYBoundary(u::Array)
    u[:, 1]
end

function upperYBoundary(u::Array)
    u[:, end]
end

function plotBoundaries(domain::Domain, u::Array)
    lx = domain.y[1]
    ux = domain.y[end]
    ly = domain.x[1]
    uy = domain.x[end]
    labels = [@sprintf("y = %5.2f", lx) @sprintf("y = %5.2f", ux) @sprintf("x = %5.2f", ly) @sprintf("x = %5.2f", uy)]
    plot([domain.y, domain.y, domain.x, domain.x], [lowerXBoundary(u), upperXBoundary(u), lowerYBoundary(u), upperYBoundary(u)], labels=labels)
end

function maximumBoundaryValue(u::Array)
    maximum([lowerXBoundary(u) upperXBoundary(u) lowerYBoundary(u) upperYBoundary(u)])
end

## ---------------------------------------- Intersection/Projection --------------------------------------------------------------

# TODO interpolate/surface projection to a plane/along a line
function interpolateAlong(x, y, u, direction, point)
    println("Not implemented yet")
end

function project(x, y, u::Array; alongX=nothing, alongY=nothing, interpolation=nothing)
    if isnothing(alongX) && isnothing(alongY)
        error("A projection method (alongX=x,alongY=y) needs to be specified.")
    end
    ax, ay = nothing, nothing
    if !isnothing(interpolation)
        U = interpolation((y, x), u)

        if !isnothing(alongX)
            ax = U(y, alongX)
        end
        if !isnothing(alongY)
            ay = U(alongY, x)
        end
    else
        #TODO throw bound error
        #Get nearest argument
        if !isnothing(alongX)
            ax = u[:, argmin(abs.(x .- alongX))]
        end
        if !isnothing(alongY)
            ay = u[argmin(abs.(y .- alongY)), :]
        end
    end
    return ax, ay
end

# Extend functionality to domains
function project(domain::Domain, u::Array; kwargs...)
    project(domain.x, domain.y, u; kwargs...)
end

#ax, ayc = project(x, y, r, alongX=2.1, alongY=y[argmin(abs.(y .- 3.5))], interpolation=cubic_spline_interpolation)

# Fine grid
#x2 = range(extrema(x)..., length=300)
#y2 = range(extrema(y)..., length=200)
# Interpolate
#z2 = [itp(x, y) for y in y2, x in x2]

## ----------------------------------------- Parameter study ----------------------------------------------------------------

function parameterStudy(study, values)
    output = similar(values)
    for i in eachindex(values)
        output[i] = study(values[i])
    end
    output
end

# ---------------------------------------- Plotting ----------------------------------------

function compareGraphs(x, numerical, analytical; kwargs...)
    plot(x, numerical; label="Numerical", kwargs...)
    plot!(x, analytical; label="Analytical", kwargs...)
end

# --------------------------- Progress diagnostic ------------------------------------------

function progress(u, prob, t)
    procentage = (t - first(prob.tspan)) / (last(prob.tspan) - first(prob.tspan)) * 100
    # Determine smallest "digits" to display unique procentage each time
    digits = ceil(Int, -log10(prob.dt / (last(prob.tspan) - first(prob.tspan)))) - 2
    println("$(round(procentage, digits=digits))% done")
end

function ProgressDiagnostic(N=100)
    Diagnostic("Progress", progress, N, "progress", assumesSpectralField=true, storesData=false)
end

# Default diagnostic
#cflDiagnostic = Diagnostic(CFLExB, 100, "cfl")
const DEFAULT_DIAGNOSTICS = [ProgressDiagnostic()]

# --------------------------------- Functionals --------------------------------------------

#1D profile: n_0(x,t) = 1/L_y∫_0^L_y n(x,y,t)dy
function radial_density_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,1], dims=1)' ./ domain.Ly
end

function poloidal_density_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,1], dims=2) ./ domain.Lx
end

function radial_vorticity_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,2], dims=1)' ./ domain.Ly
end

function poloidal_vorticity_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,2], dims=2) ./ domain.Lx
end

#Γ_0(x,t) = 1/L_y∫_0^L_y nv_x dy
function radial_flux_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    v_x, v_y = vExB(u, domain)
    sum(u[:,:,1].*v_x, dims=1)'/domain.Ly
end

#Γ_0(t) = 1/(L_xL_y)∫_0^L_x∫_0^L_y nv_x dydx
# TODO implement quadrature and make it more efficient
function radial_flux(u::AbstractArray{<:Number}, prob::SpectralODEProblem, t::Number; quadrature=nothing)
    v_x, v_y = vExB(u, prob.domain)
    sum(u[:,:,1].*v_x)/(prob.domain.Lx*prob.domain.Ly)
end

function RadialFluxDiagnostic(N = 10)
    Diagnostic("Radial flux", radial_flux, N, "radial flux")
end

# Energy integrals 
# P(t) = ∫dx 1/2n^2
function potential_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    sum(u[:,:,1].^2)/2.0
end

function PotentialEnergyDiagnostic(N = 10)
    Diagnostic("Potential energy integral", potential_energy_integral, N, "potential energy")
end

# K(t) = ∫1/2(∇_⟂Φ)^2 = ∫dx1/2 U_E^2
function kinetic_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    v_x, v_y = vExB(u, domain)
    sum(v_x.^2 .+ v_y.^2)/2.0 
end

function KineticEnergyDiagnostic(N = 10)
    Diagnostic("Kinetic energy integral", kinetic_energy_integral, N, "kinetic energy")
end

# ---------------------------------- Other -------------------------------------------------
