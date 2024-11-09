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
using Profile
using BenchmarkTools

## Run scheme test for Burgers equation
domain = Domain(128, 1)
u0 = gaussian.(domain.x', domain.y, 1, 0, 0.08) #initial_condition(gaussian, domain, l=0.08)
u1 = initial_condition(gaussian, domain, l=0.9)


plan = plan_rfft(u0)
iplan = plan_irfft(p * u0, domain.Ny)

function fftTest(u, p, ip)
    A = ip * (p * u)
    maximum(A - irfft(rfft(u), domain.Ny))
end

@profview fftTest(u1, p, ip)

@benchmark ip * (p * rand(size(u0)...))
@benchmark irfft(rfft(rand(size(u0)...)), domain.Ny)

##
surface(irfft(u0_hat, domain.Ny))
surface(u0)
surface(u1)

typeof(ip)

u0_hat = p*u0
t = Tuple([-N÷4+1:N+N÷4 for N in size(u0_hat)])
using PaddedViews

@time ifftshift(PaddedView(0, fftshift(u0_hat), t)[t...])

using Plots
heatmap(log10.(norm.(u0_hat)))#, color = :warm)

plotFrequencies(U)