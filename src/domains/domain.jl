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
    SOC<:SpectralOperators.SpectralOperatorCache,TP<:AbstractTransformPlans,T<:AbstractFloat} <: AbstractDomain

    Nx::Int
    Ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    x::X
    y::Y
    kx::KX
    ky::KY
    SC::SOC
    transforms::TP
    use_cuda::Bool
    precision::DataType
    real_transform::Bool
    dealiased::Bool
    nfields::Int

    # TODO rethink constructors interface
    Domain(N) = Domain(N, N, Lx=1, Ly=1)
    function Domain(Nx, Ny; Lx=1, Ly=1, use_cuda=true, precision=Float64, real_transform=true,
        dealiased=true, x0=-Lx / 2, y0=-Ly / 2, nfields=3)

        # Compute step sizes
        dx = Lx / Nx
        dy = Ly / Ny

        # dx and dy is subtracted at the end, because of periodic boundary conditions
        x = LinRange{precision}(x0, x0 + Lx - dx, Nx)
        y = LinRange{precision}(y0, y0 + Ly - dy, Ny)

        # Check that CUDA is compatible
        if use_cuda && !CUDA.functional()
            use_cuda = false
            @warn("CUDA is not functional")
        end

        # Prepare frequencies
        kx, ky = prepare_frequencies(Nx, Ny, dx, dy, use_cuda, precision, real_transform)

        # Prepare transform plans
        transform_plans = prepare_transform_plans(Nx, Ny, use_cuda, precision, real_transform)

        # Prepare spectral operator cache
        SC = SpectralOperators.SpectralOperatorCache(kx, ky, Nx, Ny, use_cuda=use_cuda,
            precision=precision, real_transform=real_transform, dealiased=dealiased)

        new{typeof(x),typeof(y),typeof(kx),typeof(ky),typeof(SC),typeof(transform_plans),
            precision}(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC,
            transform_plans, use_cuda, precision, real_transform, dealiased, nfields)
    end
end

# Helpers
function prepare_frequencies(Nx, Ny, dx, dy, use_cuda, precision, real_transform)

    # Compute frequencies, impose Hermitian symetry if real_transform
    kx = 2 * π * fftfreq(Nx, 1 / dx)
    ky = 2 * π * (real_transform ? rfftfreq(Ny, 1 / dy) : fftfreq(Ny, 1 / dy))

    # Enforce precision
    kx = Vector{precision}(kx)
    ky = Vector{precision}(ky)

    # TODO make more generalized, perhaps mem=CuArray
    # Transfer wave numbers to GPU if needed
    if use_cuda
        kx = adapt(CuArray, kx)
        ky = adapt(CuArray, ky)
    end

    return kx, ky
end

function prepare_transform_plans(Nx, Ny, use_cuda, precision, real_transform)

    # Temporarly create an array to create the transform plan
    utmp = use_cuda ? CUDA.zeros(precision, Ny, Nx) : zeros(precision, Ny, Nx)

    if real_transform
        FT = plan_rfft(utmp)
        iFT = plan_irfft(FT * utmp, Ny)
        return rFFTPlans(FT, iFT)
    else
        return FFTPlans(plan_fft(utmp), plan_ifft(utmp))
    end
end

# ----------------------------------- Interface --------------------------------------------

function Base.show(io::IO, m::MIME"text/plain", d::AbstractDomain)
    typename = nameof(typeof(d))

    if get(io, :compact, false)
        print(io, typename, "(", d.Nx, ",", d.Ny, ",", d.Lx, ",", d.Ly, ")")
    else
        print(io, typename, "(Nx:", d.Nx, ", Ny:", d.Ny, ", Lx:", d.Lx, ", Ly:", d.Ly,
            ", real_transform:", d.real_transform, ", dealiased:", d.dealiased, ", CUDA:",
            d.use_cuda, ")")
        if first(d.x) != 0.0 || first(d.y) != 0.0
            print(io, " offset by (", first(d.x), ", ", first(d.y), ")")
        end
    end
end

# Getters
get_transform_plans(domain::AbstractDomain) = domain.transforms
get_fwd(domain::AbstractDomain) = SpectralOperators.get_fwd(get_transform_plans(domain))
get_bwd(domain::AbstractDomain) = SpectralOperators.get_bwd(get_transform_plans(domain))
get_precision(domain::AbstractDomain) = domain.precision
get_lengths(domain::AbstractDomain) = (domain.Lx, domain.Ly)

# Aliases
const fwd = get_fwd
const bwd = get_bwd

# TODO add docstrings
spectral_size(domain::AbstractDomain) = size(get_bwd(domain))
area(domain::AbstractDomain) = domain.Lx * domain.Ly

# Overloading
Base.size(domain::AbstractDomain) = (domain.Nx, domain.Ny)
Base.length(domain::AbstractDomain) = prod(size(domain))

# ----------------------------------- Operators --------------------------------------------

# Allows spectralOperators to be called using the domains

function diff_x(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diff_x(field, domain.SC)
end

function diff_y(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diff_y(field, domain.SC)
end

function diff_xx(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diff_xx(field, domain.SC)
end

function diff_yy(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.diff_yy(field, domain.SC)
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
    SpectralOperators.quadratic_term(u, v, domain.SC)
end

function poisson_bracket(A::U, B::V, domain::D) where {U<:AbstractArray,V<:AbstractArray,
    D<:AbstractDomain}
    SpectralOperators.poisson_bracket(A, B, domain.SC)
end

function solve_phi(field::F, domain::D) where {F<:AbstractArray,D<:AbstractDomain}
    SpectralOperators.solve_phi(field, domain.SC)
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