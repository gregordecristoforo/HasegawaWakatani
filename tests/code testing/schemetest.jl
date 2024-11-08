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
u0 = Gaussian.(domain.x', domain.y, 1, 0, 0.08)

function f1(u, d, p, t)
    0
end

function f2(u, d, p, t)
    return zeros(size(u))#-diffX(u, d)
end

# Parameters
parameters = Dict(
    "nu" => 0.001,
    "g" => 1
)

t_span = [0, 10]
dt = 1

prob = SpectralODEProblem(f2, domain, u0, t_span, p=parameters, dt=dt)
tend, uend = spectral_solve(prob, MSS3())

surface(domain, uend)

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



prob.dt

mutable struct MSS3Cache
    #Coefficents
    u::AbstractArray
    c::AbstractArray
    u0::AbstractArray
    u1::AbstractArray
    u2::AbstractArray
    k0::AbstractArray
    k1::AbstractArray
    tab::AbstractTableau
    step::Integer
end
