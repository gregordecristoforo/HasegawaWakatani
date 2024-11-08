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

    domain = Domain(128, 128, 1, 1, real=false)
    #Domain(128, 1)

    phi = Gaussian.(domain.x', domain.y, A, B, value)
    phi_hat = fft(phi) #rfft(phi)
    omega_hat = laplacian(phi_hat, domain)
    omega = real(ifft(omega_hat))#irfft(omega_hat, domain.Nx)
    analytical = Omega(domain.x', domain.y, A=A, l=value)

    return norm(analytical .- omega) / length(omega)
end

studyLVariation(0.10)

ls = 0.005:0.01:1
values = parameterStudy(studyLVariation, ls)
ls[argmax(values)]

display(plot(ls, values, xlabel=L"l", ylabel=L"||\Omega-\nabla^2_\perp\phi_{num}||/N^2",
    title=L"l" * "-variational study", label=""))
savefig("tests/code testing/figures/l-variational study.pdf")

## Test 1.1
B = 0
l = 0.39
phi = Gaussian.(domain.x', domain.y, A, B, l)
surface(domain, phi)

# Plot the boundaries
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

compareGraphs(domain.x[8:120], project(domain, omega, alongY=0,
        interpolation=cubic_spline_interpolation)[2][8:120],
    project(domain, analytical, alongY=0,
        interpolation=cubic_spline_interpolation)[2][8:120],
    xlabel=L"x", ylabel=L"\Omega(x,0)",
    title=L"\Omega(x,0)" * ", " * L"(l=0.39)")
savefig("tests/code testing/figures/comparingOmegaL=0.39.pdf")

# Calculate derivative along X
dxphi_hat = diffX(phi_hat, domain)
dxphi = irfft(dxphi_hat, domain.Ny)
plotlyjsSurface(z=dxphi)

# Calculate derivative along Y
dyphi_hat = diffY(phi_hat, domain)
dyphi = irfft(dyphi_hat, domain.Ny)
plotlyjsSurface(z=dxphi .+ dyphi)

## Test 1.2
B = 1
l = 0.25
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

# Calculate derivative along X
dxphi_hat = diffX(phi_hat, domain)
dxphi = irfft(dxphi_hat, domain.Ny)
plotlyjsSurface(z=dxphi)

# Calculate derivative along Y
dyphi_hat = diffY(phi_hat, domain)
dyphi = irfft(dyphi_hat, domain.Ny)
plotlyjsSurface(z=dxphi .+ dyphi)

display(compareGraphs(domain.x[8:120], project(domain, omega, alongY=0,
        interpolation=cubic_spline_interpolation)[2][8:120],
    project(domain, analytical, alongY=0,
        interpolation=cubic_spline_interpolation)[2][8:120],
    xlabel=L"x", ylabel=L"\Omega(x,0)",
    title=L"\Omega(x,0)" * ", " * L"(l=5)"))
savefig("tests/code testing/figures/comparingOmegaL=$l.pdf")

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
savefig("tests/code testing/figures/OmegaNumericalL=0.08.pdf")

# Analytical
analytical = Omega(domain.x', domain.y, A=A, l=l)
display(plot(domain, analytical, st=:surface, xlabel="x", ylabel="y", title="Analytical Ω"))
savefig("tests/code testing/figures/OmegaAnalyticalL=0.08.pdf")

display(plot(domain, analytical .- omega, st=:surface))

## Test 2.1

function coefficents(k_x, k_y, A, B, l)
    if k_x == 0 && k_y == 0
        return 0
    end
    k_perp = k_x^2 + k_y^2
    return (A * exp(-l^2 * k_perp / 2)) / (-k_perp * 2 * l)
end

function term(x, y, k_x, k_y, A, B, l)
    if k_x == 0 && k_y == 0
        return 0
    end
    k_perp = k_x^2 + k_y^2
    return ((A * l * exp(-l^2 * k_perp / 2)) / (-2 * k_perp)) * exp(im * (k_x * x + k_y * y))
end

l = 0.08
A = 2

vals = zeros(size(phi))
for k_x in domain.kx
    for k_y in domain.kx
        vals += term.(domain.x', domain.y, k_x, k_y, A, B, l)
    end
end

vals = real(vals)
plotlyjsSurface(z=vals)

#phi_hat = Matrix{ComplexF64}([coefficents(k_x, k_y, A, B, l) for k_y in domain.ky, k_x in domain.kx])
#analytical = irfft(phi_hat, domain.Ny)
#analytical = fftshift(analytical)

#omega_hat2 = laplacian(phi_hat, domain)
#omega2 = irfft(omega_hat2, domain.Ny)

#plotlyjsSurface(z=analytical)

l = 0.08
A = 2
B = 0
omega = Gaussian.(domain.x', domain.y, A, B, l)
omega_hat = rfft(omega)
phi_hat = real(solvePhi(omega_hat, domain))
phi = irfft(phi_hat, domain.Ny)

plotlyjsSurface(z=vals .- phi)
plotlyjsSurface(z=phi)

##
import FFTW.irfft
# Possibly unstable
function irfft(matrix::Matrix)
    irfft(matrix, size(matrix)[end])
end