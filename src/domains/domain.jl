include("../operators/spectralOperators.jl")
using .SpectralOperators

abstract type AbstractDomain end

# Assumed 1st direction uses rfft, while all others use fft
"""
    Domain(N, L)
    Domain(Nx, Ny, Lx, Ly)

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
    SOC<:SpectralOperators.SpectralOperatorCache,TP<:TransformPlans} <: AbstractDomain

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

        SC = SpectralOperators.SpectralOperatorCache(kx, ky, Nx, Ny, realTransform=realTransform,
            anti_aliased=anti_aliased)

        new{typeof(x),typeof(y),typeof(kx),typeof(ky),typeof(SC),
            typeof(transform_plans)}(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC,
            transform_plans, realTransform, anti_aliased, nfields)
    end
end

# Allows spectralOperators to be called using the domains

function diff_x(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diffX(field, domain.SC)
end

function diff_y(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diffY(field, domain.SC)
end

function diff_xx(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diffXX(field, domain.SC)
end

function diff_yy(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diffYY(field, domain.SC)
end

function laplacian(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.laplacian(field, domain.SC)
end

const Δ = laplacian
const diffusion = laplacian

function hyper_diffusion(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.hyper_diffusion(field, domain.SC)
end

function quadratic_term(u::U, v::V, domain::D) where {U<:AbstractArray,V<:AbstractArray,
    D<:AbstractDomain}
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    SpectralOperators.quadraticTerm(u, v, domain.SC)
end

function poisson_bracket(A::U, B::V, domain::D) where {U<:AbstractArray,V<:AbstractArray,
    D<:AbstractDomain}
    SpectralOperators.poissonBracket(A, B, domain.SC)
end

function solve_phi(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.solvePhi(field, domain.SC)
end

function reciprocal(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.spectral_function(u -> div(1, u), field, domain.SC)
end

function spectral_exp(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.spectral_function(exp, field, domain.SC)
end

function spectral_expm1(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.spectral_function(expm1, field, domain.SC)
end

function spectral_log(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.spectral_function(log, field, domain.SC)
end