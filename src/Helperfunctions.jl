module Helperfunctions
export getCFL, energyIntegral, probe

using LinearAlgebra
using Plots

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

"""
"""

function HeatEquationAnalyticalSolution(n0, D, K, t):
    @. n0*exp(D*K*t)

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A-B))
    plot(x,A)
    #plot(x,x,B)
end

end