## Run all (alt+enter)
using Plots
include("../../src/domain.jl")
using .Domains
include("../../src/diagnostics.jl")
include("../../src/utilities.jl")
using LinearAlgebra
using LaTeXStrings

# Omega calculated from Ω = ∇²⟂ϕ, assuming Gaussian ϕ    
function Omega(x, y; A=1, l=1)
    p = (x .^ 2 .+ y .^ 2) / (2 * l^2)
    @. 2 * A / l^2 * (p - 1) * exp(-p)
end

domain = Domain(128, 1)

A = 2

# Create a l variable study
function studyLVariation(value, B=0)
    A = 2

    domain = Domain(128, 1)

    phi = Gaussian.(domain.x', domain.y, A, B, value)
    phi_hat = rfft(phi)
    omega_hat = laplacian(phi_hat, domain)
    omega = irfft(omega_hat, domain.Nx)
    analytical = Omega(domain.x', domain.y, A=A, l=value)

    return norm(analytical .- omega) / length(omega)
end

studyLVariation(0.10)

ls = 0.01:0.01:5
values = parameterStudy(studyLVariation, ls)
ls[argmax(values)]

plot(ls, values, xlabel=L"l", ylabel=L"||\Omega-\nabla^2_\perp\phi_{num}||/N^2" , title=L"l"*"-variational study",label="")
savefig("tests/code testing/figures/l-variational study.pdf")

## Test 1.1
B = 0
l = 0.39
phi = Gaussian.(domain.x', domain.y, A, B, l)
surface(domain, phi)

display(plotBoundaries(domain, phi))
print(maximumBoundaryValue(phi))

# Calculate omega
phi_hat = rfft(phi)
omega_hat = laplacian(phi_hat, domain)
omega = irfft(omega_hat, domain.Nx)
display(plot(domain, omega, st=:surface, xlabel="x", ylabel="y", title="Numerical Ω"))

# Analytical
analytical = Omega(domain.x', domain.y, A=A, l=l)
display(plot(domain, analytical, st=:surface, xlabel="x", ylabel="y", title="Analytical Ω"))

display(plot(domain, analytical .- omega, st=:surface))

## Test 1.2
B = 1
l = 0.08
phi = Gaussian.(domain.x', domain.y, A, B, l)
surface(domain, phi)

display(plotBoundaries(domain, phi))
print(maximumBoundaryValue(phi))

# Calculate omega
phi_hat = rfft(phi)
omega_hat = laplacian(phi_hat, domain)
omega = irfft(omega_hat, domain.Nx)
display(plot(domain, omega, st=:surface, xlabel="x", ylabel="y", title="Numerical Ω"))

# Analytical
analytical = Omega(domain.x', domain.y, A=A, l=l)
display(plot(domain, analytical, st=:surface, xlabel="x", ylabel="y", title="Analytical Ω"))

display(plot(domain, analytical .- omega, st=:surface))

plotlyjsSurface(z=analytical)
## Test 1.3
B = 10
l = 0.08
phi = Gaussian.(domain.x', domain.y, A, B, l)
surface(domain, phi)

display(plotBoundaries(domain, phi))
print(maximumBoundaryValue(phi))

# Calculate omega
phi_hat = rfft(phi)
omega_hat = laplacian(phi_hat, domain)
omega = irfft(omega_hat, domain.Nx)
display(plot(domain, omega, st=:surface, xlabel="x", ylabel="y", title="Numerical Ω"))

# Analytical
analytical = Omega(domain.x', domain.y, A=A, l=l)
display(plot(domain, analytical, st=:surface, xlabel="x", ylabel="y", title="Analytical Ω"))

display(plot(domain, analytical .- omega, st=:surface))





##
import FFTW.irfft
# Possibly unstable
function irfft(matrix::Matrix)
    irfft(matrix, size(matrix)[end])
end