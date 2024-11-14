## Run all (alt+enter)
using Plots
include("../src/domain.jl")
using .Domains
include("../src/diagnostics.jl")
include("../src/utilities.jl")
using LinearAlgebra
using LaTeXStrings
include("../src/spectralODEProblem.jl")
include("../src/schemes.jl")
include("../src/spectralSolve.jl")

## Run scheme test for Burgers equation
domain = Domain(1, 512, 1, 14) #domain = Domain(64, 14)
u0 = initial_condition(gaussian, domain)

# Burgers equation 
function f(u, d, p, t)
    zero(u)
end

# Parameters
parameters = Dict(
    "nu" => 0#0.01
)

t_span = [0, 52]

prob = SpectralODEProblem(f, domain, u0, t_span, p=parameters, dt=0.001)

## Solve and plot
tend, uend = spectral_solve(prob, MSS3())

plot(domain.y, uend, xlabel="x", ylabel="y")
