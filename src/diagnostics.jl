using LinearAlgebra
using Plots
using FFTW
using Printf
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

mutable struct Diagnostics
    method::Function
    sampleStep::Integer
    label::Any
    data::AbstractArray

    function Diagnostics(method::Function, sampleStep::Integer, label)
        # TODO add initialization of diagnostic somehow
        new(method, sampleStep, label, Vector[])
    end
end

function initializeDiagnostic!(diagnostic::Diagnostics, prob) #::SpectralODEProblem
    # Extract values
    tend = last(prob.tspan)
    dt = prob.dt

    # Allocate data for fields
    N = floor(Int, tend / dt / diagnostic.sampleStep)

    # Take diagnostic of initial field
    id = diagnostic.method(prob.u0, prob, first(prob.tspan))

    diagnostic.data = Vector{typeof(id)}(undef, N)
    diagnostic.data[1] = id
end

# fields, radial velocity

# --------------------------------- Probe --------------------------------------------------

# probe/high time resolution (fields, velocity etc...)

function probe(u::AbstractArray, domain::Domain, x::Number, y::Number, interpolation=nothing)
    if isnothing(interpolation)
        i = argmin(abs.(domain.x .- x))
        j = argmin(abs.(domain.y .- y))
        return u0[j, i]
    else
        U = interpolation((domain.y, domain.x), u)
        return U(y, x)
    end
end

function probe(u::AbstractArray, domain::Domain, xs::AbstractArray, ys::AbstractArray, interpolation=nothing)
    if size(xs) != size(ys)
        throw("Size of xs and ys needs to match")
    end
    data = similar(xs)
    for i in eachindex(xs)
        data[i] = probe(u, domain, xs[i], ys[i], interpolation)
    end
    return data
end

function probe(u::AbstractArray, prob, t::Number)
    u[500,1]
end

probeDiagnostic = Diagnostics(probe, 10, "probe")

# --------------------------------------- Other --------------------------------------------

#1D profile: n_0(x,t) = 1/L_y∫_0^L_y n(x,y,t)dy
#          : Γ_0(x,t) = 1/L_y∫_0^L_y nv_x dy


"""
Empty
"""

# energy integrals 

# P(t) = ∫dx 1/2n^2
# K(t) = ∫1/2(∇_⟂Φ)^2 = ∫dx1/2 U_E^2
function energyIntegral()
    nothing
end

# outputCenterOfMass::Bool

function plotFrequencies(u)
    heatmap(log10.(norm.(u)), title="Frequencies")
end
#---------------------------- Display diagnostic -------------------------------------------

function displayField(u::AbstractArray, prob, t::Number)
    display(plot(u))
end

displayFieldDiagnostic = Diagnostics(displayField, 1000, "Display")

#---------------------------------- CFL ----------------------------------------------------

# Calculate velocity assuming U_ExB = ̂z×∇Φ   
function vExB(u::AbstractArray, domain::Domain)
    W_hat = u[:, :, 2] #Assumption
    phi_hat = solvePhi(W_hat, domain)
    irfft(-diffY(phi_hat, domain), domain.Ny), irfft(diffX(phi_hat, domain), domain.Ny)
end

#Returns max CFL
function CFLExB(u::AbstractArray, prob, t::Number)
    v_x, v_y = vExB(u, prob.domain)
    #(CFLx, CFLy, x, y)
    maximum(v_x) * prob.dt / prob.domain.dx, maximum(v_y) * prob.dt / prob.domain.dy
end

function maxCFLx(u::AbstractArray, domain::Domain, dt::Number, v::Function=vExB)
    v_x, v_y = v(u, domain)
    maximum(v_x) * dt / domain.dx, argmax(v_x)
end

function maxCFLy(u::AbstractArray, domain::Domain, dt::Number, v::Function=vExB)
    v_x, v_y = v(u, domain)
    maximum(v_y) * dt / domain.dy, argmax(v_y)
end

# CFL where field is velocity
function burgerCFL(u::AbstractArray, prob, t::Number)
    maximum(u) * prob.dt / prob.domain.dy
end

burgerDiagnostic = Diagnostics(burgerCFL, 100, "CFL")

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
using Interpolations

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

x = range(-2, 3, length=5)
y = range(3, 4, length=5)
z = @. cos(x') + sin(y)

r = rand(size(z)...)

# Fine grid
x2 = range(extrema(x)..., length=300)
y2 = range(extrema(y)..., length=200)
# Interpolate
#z2 = [itp(x, y) for y in y2, x in x2]
# Plot

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

#n = u[:, :, 1]
#W = u[:, :, 2]
#println(size(n), size(W))
#display(surface(domain.x, domain.y, n, xlabel="x", ylabel="y"))
#display(contourf(W))
#display(plot(domain.y, real(multi_ifft(cache.u, domain.transform)), title="t=$t, cfl=$cfl"))
#display(contourf(domain, real(multi_ifft(cache.u, domain.transform)), title="t=$t, cfl=$cfl"))

# Default diagnostic
cflDiagnostic = Diagnostics(CFLExB, 100, "cfl")
default_diagnostics = [cflDiagnostic]