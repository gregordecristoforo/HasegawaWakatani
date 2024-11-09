using Plots
include("../../src/domain.jl")
using .Domains
include("../../src/diagnostics.jl")
include("../../src/utilities.jl")
using FFTW

using LinearAlgebra
using LaTeXStrings

## Test implementation of vPhi

function vPhi(phi::AbstractArray, domain::Domain)
    -diffY(phi, domain), diffX(phi, domain)
end

function cflPhi(phi::AbstractArray, domain::Domain, dt)
    v_x, v_y = vPhi(phi, domain)
    maximum(irfft(v_x, domain.Ny))*dt/domain.dx, maximum(irfft(v_y, domain.Ny))*dt/domain.dx
end

domain = Domain(128, 1)
phi = initial_condition(sinusoidal, domain)
plot(domain, phi)
n0 = initial_condition(gaussian,domain,l=0.08)
plot(domain, n0)

phi_hat = rfft(phi)
v_x, v_y = vPhi(phi_hat, domain)

surface(irfft(v_y, domain.Ny))

cflPhi(phi_hat, domain, 0.0006217)


function multi_irfft()
    irfft()
end