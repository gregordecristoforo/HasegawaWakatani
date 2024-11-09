## Run all (alt+enter)
using Plots
include("../../src/domain.jl")
using .Domains
include("../../src/diagnostics.jl")
include("../../src/utilities.jl")
using LinearAlgebra
using LaTeXStrings
include("../../src/spectralODEProblem.jl")
include("../../src/schemes.jl")
include("../../src/spectralSolve.jl")

## Run scheme test for Burgers equation
domain = Domain(128, 1)
u0 = gaussian.(domain.x', domain.y, 1, 0, 0.08)

function f1(u, d, p, t)
    n = u[:, :, 1]
    [-poissonBracket(n, p["phi"], d);;; zeros(size(n))]
end

function f2(u, d, p, t)
    return -0.01 * quadraticTerm(u, diffX(u, d))
end

function f3(u, d, p, t)
    poissonBracket(u, solvePhi(u, d), d)
end

# Parameters
parameters = Dict(
    "nu" => 0.01,
    "g" => 1,
    "phi" => rfft(sinusoidal.(domain.x', domain.y))
)

t_span = [0, 0.01]

prob = SpectralODEProblem(f1, domain, [u0;;; u0], t_span, p=parameters, dt=0.0001)

using BenchmarkTools

## Solve and plot

@time tend, uend2 = spectral_solve(prob, MSS3())

surface(domain, uend2[:, :, 1])
xlabel!("x")

plotlyjsSurface(z=sinusoidalX.(domain.x', domain.y, 1, 1))
plotlyjsSurface(z=uend2[:, :, 1])
contourf(domain, uend2[:, :, 1], xlabel="x", ylabel="y")

## Check caching
cache2 = get_cache(prob, MSS2())
u1 = perform_step!(cache2, prob, 0)
u1 = perform_step!(cache2, prob, 1 * prob.dt)
u1 = perform_step!(cache2, prob, 2 * prob.dt)
cache3 = get_cache(prob, MSS3())
u2 = perform_step!(cache3, prob, 0)
u2 = perform_step!(cache3, prob, prob.dt)
u2 = perform_step!(cache3, prob, 2 * prob.dt)

u1 = cache2.u0
u2 = cache3.u1
maximum(irfft(u1, domain.Ny) - irfft(u2, domain.Ny))

surface(irfft(u1, domain.Ny))
surface(irfft(u2, domain.Ny))