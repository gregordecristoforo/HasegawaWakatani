module Helperfunctions
export getCFL, energyIntegral, probe, ifftPlot

using LinearAlgebra
using Plots
using FFTW

"""
Returns max courant number at certain index
v - velocity field
Δx - spatial derivative
Δt - timestep
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
end
