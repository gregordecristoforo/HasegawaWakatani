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
struct Domain{X<:AbstractArray,Y<:AbstractArray,KX<:AbstractArray,KY<:AbstractArray,
    SOC<:SpectralOperatorCache,TP<:TransformPlans}

    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::X
    y::Y
    kx::KX
    ky::KY
    SC::SOC
    transform::TP
    realTransform::Bool
    anti_aliased::Bool
    nfields::Int

    Domain(N) = Domain(N, 1)
    Domain(N, L) = Domain(N, N, L, L)
    function Domain(Nx, Ny, Lx, Ly; realTransform=true, anti_aliased=false, x0=-Lx / 2, y0=-Ly / 2, nfields=3)
        dx = Lx / Nx
        dy = Ly / Ny
        # dx and dy is subtracted at the end, because of periodic boundary conditions
        x = LinRange(x0, x0 + Lx - dx, Nx)
        y = LinRange(y0, y0 + Ly - dy, Ny)

        # ------------------ If x-direction favored in rfft -------------------
        #if Nx > Ny
        #    kx = realTransform ? 2 * π * rfftfreq(Nx, 1 / dx) : 2 * π * fftfreq(Nx, 1 / dx)
        #    ky = 2 * π * fftfreq(Ny, 1 / dy)
        #else
        kx = 2 * π * fftfreq(Nx, 1 / dx)
        ky = realTransform ? 2 * π * rfftfreq(Ny, 1 / dy) : 2 * π * fftfreq(Ny, 1 / dy)

        utmp = zeros(Float64, Ny, Nx)

        if realTransform
            FT = plan_rfft(utmp)
            iFT = plan_irfft(FT * utmp, Ny)
            transform_plans = rFFTPlans(FT, iFT)
        else
            transform_plans = FFTPlans(plan_fft(utmp), plan_ifft(utmp))
        end

        SC = SpectralOperatorCache(kx, ky, Nx, Ny, realTransform=realTransform,
            anti_aliased=anti_aliased)

        new{typeof(x),typeof(y),typeof(kx),typeof(ky),typeof(SC),
            typeof(transform_plans)}(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC,
            transform_plans, realTransform, anti_aliased, nfields)
    end
end

# Allow spectralOperators to be called using the domains

# TODO add snake_case
function diffX(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.diffX(field, domain.SC)
end

function diffY(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.diffY(field, domain.SC)
end

function diffXX(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.diffXX(field, domain.SC)
end

function diffYY(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.diffYY(field, domain.SC)
end

function laplacian(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.laplacian(field, domain.SC)
end

const Δ = laplacian
const diffusion = laplacian

function hyper_diffusion(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.hyper_diffusion(field, domain.SC)
end

function quadraticTerm(u::U, v::V, domain::D) where {U<:AbstractArray,V<:AbstractArray,D<:Domain}
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    SpectralOperators.quadraticTerm(u, v, domain.SC)
end

function poissonBracket(A::U, B::V, domain::D) where {U<:AbstractArray,V<:AbstractArray,D<:Domain}
    SpectralOperators.poissonBracket(A, B, domain.SC)
end

function solvePhi(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.solvePhi(field, domain.SC)
end

function reciprocal(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.spectral_function(u -> div(1, u), field, domain.SC)
end

function spectral_exp(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.spectral_function(exp, field, domain.SC)
end

function spectral_expm1(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.spectral_function(expm1, field, domain.SC)
end

function spectral_log(field::F, domain::D) where {F<:AbstractArray,D<:Domain}
    SpectralOperators.spectral_function(log, field, domain.SC)
end

export Domain, diffX, diffXX, diffY, diffYY, poissonBracket, solvePhi, quadraticTerm,
    diffusion, laplacian, Δ, SpectralOperatorCache, reciprocal, spectral_exp, spectral_expm1,
    spectral_log, hyper_diffusion
end