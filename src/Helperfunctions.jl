module Helperfunctions
export Domain, getDomainFrequencies, getCFL, energyIntegral, probe, ifftPlot, testTimestepConvergence, testResolutionConvergence

using LinearAlgebra
using Plots
using FFTW

"""
Box domain, that calculates spatial resolution under construction.

# Contains
Lengths: ``Lx``, ``Ly`` (Float64)\\
Number of grid point: ``Nx``, ``Ny`` (Int64)\\
Spatial resolution: ``dx``, ``dy`` (Float64)\\
Spatial points: ``x``, ``y`` (LinRange)

``dxᵢ = 2Lₓ÷(Nₓ-1)``

Square Domain can be constructed using:\\
``Domain(N,L)``

Rectangular Domain can be constructed using:\\
``Domain(Nx,Ny,Lx,Ly)``
"""
struct Domain
    Nx::Int64
    Ny::Int64
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    Domain(N) = Domain(N, 1)
    Domain(N, L) = Domain(N, N, L, L)
    function Domain(Nx, Ny, Lx, Ly)
        dx = 2 * Lx / (Nx - 1)
        dy = 2 * Ly / (Nx - 1)
        x = LinRange(-Lx, Lx, Nx)
        y = LinRange(-Ly, Ly, Ny)
        new(Nx, Ny, Lx, Ly, dx, dy, x, y)
    end
end

function getDomainFrequencies(domain::Domain)
    k_x = 2 * π * fftfreq(domain.Nx, 1 / domain.dx)
    k_y = 2 * π * fftfreq(domain.Ny, 1 / domain.dy)
    return k_x, k_y
end

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

function probe(x, y, t, type="Interpolate")
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
function testTimestepConvergence(method, initialField, timesteps)
    D = 1.0
    domain = Domain(256, 4)
    k_x, k_y = getDomainFrequencies(domain)
    K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]
    tend = 1

    u0 = initialField.(domain.x, domain.y')
    analyticalSolution = HeatEquationAnalyticalSolution(u0, D, K, tend)

    residuals = zeros(size(timesteps))
    du = similar(u0)

    for (i, dt) in enumerate(timesteps)
        #method(Laplacian, initialField)
        du = method(du, u0, K, dt, tend)
        #method(fun, t_span, dt, n0, p)
        residuals[i] = norm(du - analyticalSolution)
    end

    plot(timesteps, residuals, xaxis=:log, yaxis=:log)
end

# Uses the Heat equation to test at the moment
function testResolutionConvergence(method, initialField, resolutions)
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

domain = Domain(4, 256)
getDomainFrequencies(domain)

end
