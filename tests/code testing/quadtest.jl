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
domain = Domain(1024, 1024, 1, 20, realTransform=false, anti_aliased=false)
domainR = Domain(1024, 1024, 1, 20, realTransform=true, anti_aliased=false)
domainA = Domain(1024, 1024, 1, 20, realTransform=false, anti_aliased=true)
domainRA = Domain(1024, 1024, 1, 20, realTransform=true, anti_aliased=true)

u0 = initial_condition(gaussianWallY, domain)

surface(u0)

# Burgers equation 
function f(u, d, p, t)
    return -quadraticTerm(u, diffY(u, d), d)
end

function test(u, domain::Domain)
    u0_hat = domain.transform.FT * u0
    real(domain.transform.iFT * f(u0_hat, domain, 0, 0))
end

plot(test(u0, domain)[:, 1])
plot!(test(u0, domainR)[:, 1])
plot(test(u0, domainA)[:, 1])
plot!(test(u0, domainRA)[:, 1])

u0_hat = fft(u0)
dudy_hat = diffY(u0_hat, domain)
dudy = ifft(dudy_hat)
nt = real(-u0 .* dudy)

plot!(nt[:, 1])

plot(domain.y, abs.(nt[:, 1]-test(u0, domain)[:, 1]))
maximum(abs.(nt[:, 1]-test(u0, domain)[:, 1]))

A = maximum(test(u0, domain)) ./ maximum(nt)
A = maximum(test(u0, domainR)) ./ maximum(nt)
A = maximum(test(u0, domainA)) ./ maximum(nt)
A = maximum(test(u0, domainRA)) ./ maximum(nt)


maximum(real(domain.SC.QTPlans.FT * u0 - fft(u0)))


u0_hat = domain.transform.FT * u0
dudy_hat = diffY(u0_hat, domain)
dudy = ifft(dudy_hat)
nt = real(-u0 .* dudy)

plot!(nt)









# Parameters
parameters = Dict(
    "nu" => 0#0.01
)

# Break down time 
dudy = diffY(domain.transform.FT * u0, domain)
t_b = -1 / (minimum(real(domain.transform.iFT * dudy)))

t_span = [0, 0.9 * t_b]
