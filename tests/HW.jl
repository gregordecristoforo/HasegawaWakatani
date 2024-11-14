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
domain = Domain(128, 128, 1, 1, anti_aliased=false)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=1, B=0, l=0.08)

function f(u, d, p, t)
    n = u[:, :, 1]
    W = u[:, :, 2]
    phi = solvePhi(W, d)
    dn = -poissonBracket(phi, n, d)
    dn += p["g"] * quadraticTerm(n, diffY(n, d), d)
    dn += -p["g"] * diffY(n, d)
    dn += -p["sigma"] * n
    dW = -poissonBracket(phi, W, d)
    dW += -p["g"] * diffY(spectral_log(n, d), d)
    #dW += -p["g"] * quadraticTerm(reciprocal(n, d), diffY(n, d), d)
    dW += n
    dW += -quadraticTerm(n, spectral_exp(phi, d), d)
    [dn;;; dW]
end

# Parameters
parameters = Dict(
    "nu" => 0.005,
    "g" => 1,
    "sigma" => 0.1,
)

t_span = [0, 100]

prob = SpectralODEProblem(f, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-5)

## Solve and plot
tend, uend = spectral_solve(prob, MSS3())

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