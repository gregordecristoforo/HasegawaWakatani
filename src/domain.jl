module Domains

using FFTW
include("spectralOperators.jl")
using .SpectralOperators

# Assumed 1st direction uses rfft, while all others use fft
"""
Box domain, that calculates spatial resolution under construction.

# Contains
Lengths: ``Lx``, ``Ly`` (Float64)\\
Number of grid point: ``Nx``, ``Ny`` (Int64)\\
Spatial resolution: ``dx``, ``dy`` (Float64)\\
Spatial points: ``x``, ``y`` (LinRange)

``dxᵢ = 2Lₓ÷(Nₓ-1)``

Square Domain can be constructed using:\\
``Domain(N,L)``

Rectangular Domain can be constructed using:\\
``Domain(Nx,Ny,Lx,Ly)``
"""
struct Domain
    Nx::Int64
    Ny::Int64
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    kx::Frequencies
    ky::Frequencies
    SC::SpectralOperatorCache
    transform::TransformPlans
    realTransform::Bool
    anti_aliased::Bool
    nfields::Integer
    Domain(N) = Domain(N, 1)
    Domain(N, L) = Domain(N, N, L, L)
    function Domain(Nx, Ny, Lx, Ly; realTransform=true, anti_aliased=false, x0=-Lx / 2, y0=-Ly / 2, nfields=3)
        dx = Lx / Nx
        dy = Ly / Ny
        # dx and dy is subtracted at the end, because periodic boundary conditions
        x = LinRange(x0, x0 + Lx - dx, Nx)
        y = LinRange(y0, y0 + Ly - dy, Ny)
        # ------------------ If x-direction favored in rfft -------------------
        #kx = real ? 2 * π * rfftfreq(Nx, 1 / dx) : 2 * π * fftfreq(Nx, 1 / dx)
        #ky = 2 * π * fftfreq(Ny, 1 / dy)
        kx = 2 * π * fftfreq(Nx, 1 / dx)
        ky = realTransform ? 2 * π * rfftfreq(Ny, 1 / dy) : 2 * π * fftfreq(Ny, 1 / dy)

        utmp = zeros(Ny, Nx)

        if realTransform
            FT = plan_rfft(utmp)
            iFT = plan_irfft(FT * utmp, Ny)
            transform_plans = rFFTPlans(FT, iFT)
        else
            transform_plans = FFTPlans(plan_fft(utmp), plan_ifft(utmp))
        end

        SC = SpectralOperatorCache(kx, ky, Nx, Ny, realTransform=realTransform,
            anti_aliased=anti_aliased)

        new(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC, transform_plans, realTransform, anti_aliased, nfields)
    end
end

# Allow spectralOperators to be called using the domains

# TODO add snake_case
function diffX(field, domain::Domain)
    SpectralOperators.diffX(field, domain.SC)
end

function diffY(field, domain::Domain)
    SpectralOperators.diffY(field, domain.SC)
end

function diffXX(field, domain::Domain)
    SpectralOperators.diffXX(field, domain.SC)
end

function diffYY(field, domain::Domain)
    SpectralOperators.diffYY(field, domain.SC)
end

function laplacian(field, domain::Domain)
    SpectralOperators.laplacian(field, domain.SC)
end

const Δ = laplacian
const diffusion = laplacian

function quadraticTerm(u, v, domain::Domain)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    SpectralOperators.quadraticTerm(u, v, domain.SC)
end

function poissonBracket(A, B, domain::Domain)
    SpectralOperators.poissonBracket(A, B, domain.SC)
end

function solvePhi(field, domain::Domain)
    SpectralOperators.solvePhi(field, domain.SC)
end

function reciprocal(field, domain::Domain)
    F = domain.transform.iFT * field
    domain.transform.FT * (F .^ (-1))
end

function spectral_exp(field, domain::Domain)
    F = domain.transform.iFT * field
    domain.transform.FT * (exp.(F))
end

function spectral_log(field, domain::Domain)
    F = domain.transform.iFT * field
    domain.transform.FT * (log.(F))
end

export Domain, diffX, diffXX, diffY, diffYY, poissonBracket, solvePhi, quadraticTerm,
    diffusion, laplacian, Δ, SpectralOperatorCache, reciprocal, spectral_exp, spectral_log

end