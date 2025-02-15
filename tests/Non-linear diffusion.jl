## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(512, 512, 14, 14, anti_aliased=false)
u0 = initial_condition(gaussian, domain; B = 1)
u0 = log.(initial_condition(gaussian, domain; B = 1))

# Diffusion 
function L(u, d, p, t)
    p["nu"]*diffusion(u, d)
end

function N1(u, d, p, t)
    p["nu"]*(quadraticTerm(diffX(u, d), diffX(u, d),d) + quadraticTerm(diffY(u, d), diffY(u, d),d)) 
end

function N2(u, d, p, t)
    -p["nu"]*(quadraticTerm(diffX(u, d), diffX(u, d),d) + quadraticTerm(diffY(u, d), diffY(u, d),d)) 
end

function N(u, d, p, t)
    zero(u)
end

# Parameters
parameters = Dict(
    "nu" => 0.01
)

t_span = [0, 52]

prob = SpectralODEProblem(L, N2, domain, u0, t_span, p=parameters, dt=0.01)

## Solve and plot
sol = spectral_solve(prob, MSS3())

plot(domain.y, u0)
plot!(domain.y, sol.u[end], xlabel="x", ylabel="y")

surface(domain, HeatEquationAnalyticalSolution(exp.(u0), domain, parameters, last(t_span)))
surface(domain, exp.(sol.u[end]), xlabel="x")

# function HeatEquationAnalyticalSolution(u0, domain, p, t)
#     u0_hat = (domain.transform.FT*u0).*exp.(p["nu"] * domain.SC.Laplacian * t)
#     domain.transform.iFT*u0_hat
# end