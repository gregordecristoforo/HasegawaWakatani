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
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(512, 512, 200, 100, anti_aliased=false, offsetX=75)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=20, B=0, l=3)

surface(domain, u0)

function f(u, d, p, t)
    n = u[:, :, 1]
    W = u[:, :, 2]
    phi = solvePhi(W, d)
    dn = -poissonBracket(phi, n, d)
    #dn += p["g"] * quadraticTerm(n, diffY(phi, d), d)
    #dn += -p["g"] * diffY(n, d)
    dn += -p["sigma"] * quadraticTerm(n, spectral_exp(3.6 .- phi, d), d)
    dW = -poissonBracket(phi, W, d)
    #dW += -p["g"] * diffY(spectral_log(n, d), d)
    dW += -p["g"] * quadraticTerm(reciprocal(n, d), diffY(n, d), d)
    dW += p["sigma"] .- p["sigma"] * spectral_exp(3.6 .- phi, d)
    [dn;;; dW]
end

# Parameters
parameters = Dict(
    "nu" => 1e-4,
    "g" => 0.01,
    "sigma" => 1e-4,
)

t_span = [0, 100]

prob = SpectralODEProblem(f, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-2)

# Solve and plot
#tend, uend = spectral_solve(prob, MSS3())
U = spectral_solve(prob, MSS3())

## Make gif
default(legend=false)
@gif for i in axes(U, 4)
    contourf(U[:, :, 2, i])
end

surface(U[:, :, 1, end])

for i in axes(U, 4)
    println(size(U[:, :, 1, i]))
end
##
surface(domain, uend)
contourf(domain, uend[:, :, 1])
xlabel!("x")

plotlyjsSurface(z=uend)
plotlyjsSurface(z=uend[:, :, 1])

## Debug

u0_hat = domain.transform.FT * u0
f_hat = f([u0_hat;;; u0_hat], domain, parameters, 0)
F = transform(f_hat, domain.transform.iFT)
plotlyjsSurface(z=F[:, :, 1])
plotlyjsSurface(z=F[:, :, 2])

plotlyjsSurface(z=(1) ./ u0)


s = domain.transform.iFT * inverse(prob.u0_hat[:, :, 1], domain)
plotlyjsSurface(z=s)