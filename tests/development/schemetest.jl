## Run all (alt+enter)
include("../../src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(1, 1, 50, 50, anti_aliased=false)
u0 = initial_condition(gaussian, domain)
u0[1] = 1

function L(u, d, p, t)
    p["lambda"]*u
end

function N(u, d, p, t)
    zero(u)#p["lambda"] * u
end

# Parameters
parameters = Dict(
    "lambda" => -1
)

t_span = [0, 10]

prob = SpectralODEProblem(L, N, domain, u0, t_span, p=parameters, dt=0.0001)

## Solve and plot
sol = spectral_solve(prob, MSS1(), Output(prob, -1, []))

function analytical_solution(u, domain, p, t)
    [u0 * exp(p["lambda"] * t);;]
end


## Time convergence test
#timesteps = [2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9, 2^-10, 2^-11, 2^-12, 2^-13, 2^-14, 2^-15, 2^-16, 2^-17, 2^-18, 2^-19, 2^-20]
timesteps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
_, convergence1 = test_timestep_convergence(prob, analytical_solution, timesteps, MSS1())
_, convergence2 = test_timestep_convergence(prob, analytical_solution, timesteps, MSS2())
_, convergence3 = test_timestep_convergence(prob, analytical_solution, timesteps, MSS3())
plot(timesteps, convergence1, xaxis=:log, yaxis=:log, label="MSS1")
plot!(timesteps, convergence2, xaxis=:log, yaxis=:log, label="MSS2", color="dark green")
plot!(timesteps, convergence3, xaxis=:log, yaxis=:log, label="MSS3", color="orange")
plot!(timesteps, 3000000 * timesteps.^ 2, linestyle=:dash, label=L"\frac{1}{2}dt^2", xlabel="dt",
    ylabel=L"||U-u_a||", title="Timestep convergence, exponential test", xticks=timesteps)
plot!(timesteps, 0.0002*timesteps, linestyle=:dash)
plot!(timesteps, 0.0001*timesteps.^2, linestyle=:dash)
plot!(timesteps, 0.0001*timesteps.^3, linestyle=:dash)
savefig("Timestep convergence, exponential test.pdf")

convergence3
## ------------------------------ Old content -----------------------------------------------

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

#FFTW.set_num_threads(8)

# Run scheme test for Burgers equation
domain = Domain(128, 128, 1, 1, anti_aliased=true)
u0 = gaussian.(domain.x', domain.y, A=1, B=0, l=0.08)

function f1(u, d, p, t)
    n = u[:, :, 1]
    [-poissonBracket(n, p["phi"], d);;; zeros(size(n))]
end

function f2(u, d, p, t)
    return -0.01 * quadraticTerm(u, diffX(u, d), d)
end

function f3(u, d, p, t)
    poissonBracket(u, solvePhi(u, d), d)
end

function f4(u, d, p, t)
    -poissonBracket(p["phi"], u, d) - p["g"] * diffY(u, d) - p["nu"] * quadraticTerm(u, diffY(p["phi"], d), d)
end

# Parameters
parameters = Dict(
    "nu" => 0.1,
    "g" => 1,
    "phi" => rfft(sinusoidal.(domain.x', domain.y))# .+ 0.25))
)

t_span = [0, 0.5]

prob = SpectralODEProblem(f1, domain, [u0;;; u0], t_span, p=parameters, dt=0.001)
prob = SpectralODEProblem(f4, domain, u0, t_span, p=parameters, dt=0.00001)

using BenchmarkTools
using Profile

## Solve and plot

tend, uend2 = spectral_solve(prob, MSS3())

surface(domain, uend2)
contourf(domain, uend2[:, :, 1])
xlabel!("x")

plotlyjsSurface(z=uend2)
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