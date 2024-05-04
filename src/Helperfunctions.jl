module Helperfunctions
export Domain, getCFL, energyIntegral, probe, ifftPlot

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
``Domain(L,N)``

Rectangular Domain can be constructed using:\\
``Domain(Lx,Ly,Nx,Ny)``
"""
struct Domain
    Lx::Float64
    Ly::Float64
    Nx::Int64
    Ny::Int64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    Domain(L, N) = Domain(L, L, N, N)
    function Domain(Lx, Ly, Nx, Ny)
        dx = 2 * Lx / (Nx - 1)
        dy = 2 * Ly / (Nx - 1)
        x = LinRange(-Lx, Lx, Nx)
        y = LinRange(-Ly, Ly, Ny)
        new(Nx, Ny, Lx, Ly, dx, dy)
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

function testTimestepConvergence(initialState, numericalScheme, analyticalSolution, timesteps)
    D = 1.0
    domain = Domain(4, 256)


    for dt in timesteps
    end
end

domain = Domain(4, 256)
getDomainFrequencies(domain)

end
