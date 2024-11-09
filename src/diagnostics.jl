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

# ----------------------------------- Diagnostics ------------------------------------------

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

# --------------------------------------- Other --------------------------------------------

function plotFrequencies(u)
    heatmap(log10.(norm.(u)), title="Frequencies")
end

#TODO implement way to get velocity of field iku?
function v(u)
end

#Calculate max cfl in x direction Pseudocode #TODO implement properly
function cflx(u)
    max(abs(v(u))) * dt / dx
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    #plot(x, A)
end

# Uses the Heat equation to test at the moment
function testTimestepConvergence(scheme, prob, analyticalSolution, timesteps)

    #Calculate analyticalSolution
    u = analyticalSolution(prob)

    #Initialize storage
    residuals = zeros(size(timesteps))

    for (i, dt) in enumerate(timesteps)
        #Change timestep of spectralODEProblem
        prob.dt = dt
        #Calculate approximate solution
        _, uN = scheme(prob, output=Nothing, singleStep=false)
        residuals[i] = norm(ifft(uN) - ifft(u))
    end

    #Plot residuals vs. time
    plot(timesteps, residuals, xaxis=:log, yaxis=:log, xlabel="dt", ylabel="||u-u_a||")
end

#
function testResolutionConvergence(scheme, prob, initialField, analyticalSolution, resolutions)
    cprob = deepcopy(prob)
    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)
        domain = Domain(N, 4)
        updateDomain!(cprob, domain)
        updateInitalField!(cprob, initialField)
        #prob = SpectralODEProblem(prob.f, domain, prob.u0, prob.tspan, p=prob.p, dt=prob.dt)
        #prob = SpectralODEProblem(prob.f, prob.domain, fft(initialField(prob.domain, prob.p)), prob.tspan, p = prob.p, dt=prob.dt)
        println(size(cprob.u0))
        u = analyticalSolution(cprob)

        _, uN = scheme(cprob, output=Nothing, singleStep=false)
        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(ifft(u) - ifft(uN)) / (domain.Nx * domain.Ny)
        println(maximum(real(ifft(u))) - maximum(real(ifft(uN))))
    end

    display(plot(resolutions, residuals, xaxis=:log2, yaxis=:log))#, st=:scatter))
    display(plot(resolutions, resolutions .^ -2, xaxis=:log2, yaxis=:log))#, st=:scatter))
end

#testResolutionConvergence(mSS1Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256])


"""
Empty
"""
#module Helperfunctions
#export Domain, getDomainFrequencies, getCFL, energyIntegral, probe, ifftPlot, testTimestepConvergence, testResolutionConvergence


"""
Returns max courant number at certain index\\
``v`` - velocity field\\
``Δx`` - spatial derivative\\
``Δt`` - timestep
"""
function getMaxCFL(v, Δx, Δt)
    CFL = v * Δt / Δx
    findmax(CFL)
end

function energyIntegral()
    nothing
end

#"""
#Checks if any of the many arguments that Plots.plot is complex 
#and if so takes the real part of the inverse Fourier transform.
#"""
"""
ifftPlot(args...; kwargs...)
Plot the real part of the inverse Fourier transform (IFFT) of each argument that is a complex array. 
This function is designed to handle multiple input arrays and plot them using the `plot` function 
from a plotting library such as Plots.jl. Non-complex arrays are plotted as-is.

# Arguments
- `args...`: A variable number of arguments. Each argument can be an array. If the array is of a complex type, 
  its IFFT is computed, and only the real part is plotted. If the array is not complex, it is plotted directly.
- `kwargs...`: Keyword arguments that are passed directly to the `plot` function to customize the plot.

# Usage
using FFTW, Plots

# Create some sample data
x = rand(ComplexF64, 100)\\
y = rand(100)

# Plot the real part of the IFFT of `x` and `y` directly
ifftPlot(x, y, title="IFFT Plot Example", legend=:topright)

"""
function ifftPlot(args...; kwargs...)
    processed_args = []
    for arg in args
        if eltype(arg) <: Complex
            arg = real(ifft(arg))
        end
        push!(processed_args, arg)
    end

    plot(processed_args...; kwargs...)
end

"""
"""
function HeatEquationAnalyticalSolution(n0, D, K, t)
    @. n0 * exp(D * K * t)
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    plot(x, A)
    #plot(x,x,B)
end

# Uses the Heat equation to test at the moment
function testResolutionConvergence(scheme, initialField, resolutions)
    D = 1.0
    dt = 0.001
    tend = 1

    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)
        domain = Domain(N, 4)
        k_x, k_y = getDomainFrequencies(domain)
        K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]

        u0 = initialField.(domain.x, domain.y')
        analyticalSolution = HeatEquationAnalyticalSolution(u0, D, K, tend)

        du = similar(u0)

        #method(Laplacian, initialField)
        du = method(du, u0, K, dt, tend)
        #method(fun, t_span, dt, n0, p)
        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(ifft(du) - ifft(analyticalSolution)) / (domain.Nx * domain.Ny)
    end

    plot(resolutions, residuals, xaxis=:log2, yaxis=:log, st=:scatter)
end