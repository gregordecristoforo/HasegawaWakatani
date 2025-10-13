# ------------------------------------- Domain ---------------------------------------------

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
    SO<:NamedTuple,TP<:AbstractTransformPlans,T<:AbstractFloat} <: AbstractDomain

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
    operators::SO
    transforms::TP
    MemoryType::Type # TODO paramaterize properly
    precision::DataType
    real_transform::Bool
    dealiased::Bool

    # TODO rethink interface
    Domain(N) = Domain(N, N, Lx=1, Ly=1)
    function Domain(Nx::Integer, Ny::Integer; Lx::Number=1, Ly::Number=1,
        MemoryType::Type{<:AbstractArray}=Array, precision::DataType=Float64,
        real_transform::Bool=true, dealiased::Bool=true, x0::Number=-Lx / 2,
        y0::Number=-Ly / 2, operators)#::Vector{Symbol}=DEFAULT_OPERATORS)

        # Compute step sizes
        dx = Lx / Nx
        dy = Ly / Ny

        # dx and dy is subtracted at the end, because of periodic boundary conditions
        x = LinRange{precision}(x0, x0 + Lx - dx, Nx)
        y = LinRange{precision}(y0, y0 + Ly - dy, Ny)

        # Prepare frequencies
        kx, ky = prepare_frequencies(Nx, Ny, dx, dy, MemoryType, precision, real_transform)

        # Prepare transform plans
        transform_plans = prepare_transform_plans(Nx, Ny, MemoryType, precision, real_transform)

        # Prepare spectral operator cache
        ops = prepare_operators(Domain, operators, (ky, kx), (Ny, Nx), MemoryType=MemoryType,
            precision=precision, real_transform=real_transform, dealiased=dealiased)

        new{typeof(x),typeof(y),typeof(kx),typeof(ky),typeof(ops),typeof(transform_plans),
            precision}(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, ops, transform_plans,
            MemoryType, precision, real_transform, dealiased)
    end
end

# Helpers
function prepare_frequencies(Nx, Ny, dx, dy, MemoryType, precision, real_transform)

    # Compute frequencies, impose Hermitian symetry if real_transform
    kx = 2 * π * fftfreq(Nx, 1 / dx) |> MemoryType{precision}
    ky = 2 * π * (real_transform ? rfftfreq(Ny, 1 / dy) : fftfreq(Ny, 1 / dy)) |> MemoryType{precision}

    return kx, ky
end

"""
    prepare_transform_plans(Nx, Ny, use_cuda, precision, real_transform)
    prepare_transform_plans(utmp, ::Type{Domain}, ::Val{false})
    prepare_transform_plans(utmp, ::Type{Domain}, ::Val{true})

  Prepares the transform plan by preparing a domain and then using dispatching to call the 
  right construction method.
"""
function prepare_transform_plans(Nx, Ny, MemoryType, precision, real_transform)

    # Temporarly create an array to create the transform plan
    utmp = zeros(precision, Ny, Nx) |> MemoryType

    # Dispatch on Domain and real_transform
    prepare_transform_plans(utmp, Domain, Val(real_transform))
end

function prepare_transform_plans(utmp, ::Type{Domain}, ::Val{true})
    FT = plan_rfft(utmp)
    iFT = plan_irfft(FT * utmp, first(size(utmp)))
    return rFFTPlans(FT, iFT)
end

function prepare_transform_plans(utmp, ::Type{Domain}, ::Val{false})
    FFTPlans(plan_fft(utmp), plan_ifft(utmp))
end

# ----------------------------------- Interface --------------------------------------------

function Base.show(io::IO, m::MIME"text/plain", d::AbstractDomain)
    typename = nameof(typeof(d))

    if get(io, :compact, false)
        print(io, typename, "(", d.Nx, ",", d.Ny, ",", d.Lx, ",", d.Ly, ")")
    else
        print(io, typename, "(Nx:", d.Nx, ", Ny:", d.Ny, ", Lx:", d.Lx, ", Ly:", d.Ly,
            ", real_transform:", d.real_transform, ", dealiased:", d.dealiased, "), mem:",
            d.MemoryType, ")")
        if first(d.x) != 0.0 || first(d.y) != 0.0
            print(io, " offset by (", first(d.x), ", ", first(d.y), ")")
        end
    end
end

# Getters
get_transform_plans(domain::AbstractDomain) = domain.transforms
get_fwd(domain::AbstractDomain) = get_fwd(get_transform_plans(domain))
get_bwd(domain::AbstractDomain) = get_bwd(get_transform_plans(domain))
get_precision(domain::AbstractDomain) = domain.precision
get_lengths(domain::AbstractDomain) = (domain.Lx, domain.Ly)

# TODO add docstrings
spectral_size(domain::AbstractDomain) = size(get_bwd(domain))
area(domain::AbstractDomain) = domain.Lx * domain.Ly

# Overloading
Base.size(domain::AbstractDomain) = (domain.Nx, domain.Ny)
Base.length(domain::AbstractDomain) = prod(size(domain))

# TODO add_alias method