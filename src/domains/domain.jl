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

    Restricted to 2D for the time being.
"""
struct Domain{X<:AbstractArray,Y<:AbstractArray,KX<:AbstractArray,KY<:AbstractArray,
    TP<:AbstractTransformPlans,T<:AbstractFloat} <: AbstractDomain

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
    real_transform::Bool
    dealiased::Bool
    transforms::TP
    MemoryType::Type
    precision::DataType

    Domain(N::Integer; L::Number=1, kwargs...) = Domain(N, N, Lx=L, Ly=L; kwargs...)
    Domain(N::Tuple{Integer}; L::Tuple{<:Number}=1, kwargs...) = Domain(N, N, Lx=L, Ly=L; kwargs...)
    function Domain(Nx::Integer, Ny::Integer; Lx::Number=1, Ly::Number=1,
        MemoryType::Type{<:AbstractArray}=Array, precision::DataType=Float64,
        real_transform::Bool=true, dealiased::Bool=true, x0::Number=-Lx / 2,
        y0::Number=-Ly / 2)

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

        new{typeof(x),typeof(y),typeof(kx),typeof(ky),typeof(transform_plans),
            precision}(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, real_transform, dealiased,
            transform_plans, MemoryType, precision)
    end
end

# Helpers
function prepare_frequencies(Nx, Ny, dx, dy, MemoryType, precision, real_transform)

    # Compute frequencies, impose Hermitian symetry if real_transform
    kx = 2 * π * fftfreq(Nx, 1 / dx) |> MemoryType{precision}
    ky = 2 * π * (real_transform ? rfftfreq(Ny, 1 / dy) : fftfreq(Ny, 1 / dy)) |> MemoryType{precision}

    return kx, ky
end

# TODO move to fftutilites.jl [#22](https://github.com/JohannesMorkrid/HasegawaWakatani.jl/issues/22)
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

# TODO make use of domain interface functions [#23](https://github.com/JohannesMorkrid/HasegawaWakatani.jl/issues/23)
function Base.show(io::IO, m::MIME"text/plain", domain::AbstractDomain)
    typename = nameof(typeof(domain))

    if get(io, :compact, false)
        print(io, typename, "(", domain.Nx, ",", domain.Ny, ",", domain.Lx, ",", domain.Ly, ")")
    else
        print(io, typename, "(Nx:", domain.Nx, ", Ny:", domain.Ny, ", Lx:", domain.Lx, ", Ly:",
            domain.Ly, ", real_transform:", domain.real_transform, ", dealiased:", domain.dealiased,
            "), MemoryType:", memory_type(domain), ")")
        if first(domain.x) != 0.0 || first(domain.y) != 0.0
            print(io, " offset by (", first(domain.x), ", ", first(domain.y), ")")
        end
    end
end

"""'
    lengths(domain::Domain)

  Returns a tuple of lengths for each axis, default (Ly, Lx).
"""
lengths(domain::Domain) = (domain.Ly, domain.Lx)

"""
    differential_elements()

  Returns a tuple of differential elements, the grid spacing along each axis, default (dy, dx).
"""
differential_elements(domain::Domain) = (domain.dy, domain.dx)

"""
    points(domain::Domain)
  
  Returns a tuple of points along each axis, default (y, x).
"""
points(domain::Domain) = (domain.y, domain.x)

"""
    wave_vectors(domain::Domain)

  Returns a tuple of wave vectors for each axis, default (ky, kx).
"""
wave_vectors(domain::Domain) = (domain.ky, domain.kx)

"""
    domain_kwargs(domain::AbstractArray)

  Returns the domain specific keyword arguments, depending on the type of AbstractDomain.
"""
domain_kwargs(domain::Domain) = (real_transform=domain.real_transform, dealiased=dealiased)

"""
    spectral_size(domain::AbstractDomain)

  Returns a tuple containing the size of the spectral coefficient Array (size in spectral space).
"""
spectral_size(domain::AbstractDomain) = size(get_bwd(domain))

"""
    area(domain::AbstractDomain)

  Computes the area of the domain. By default uses prod(lengths(domain)).
"""
area(domain::AbstractDomain) = prod(lengths(domain))

# Getters
get_transform_plans(domain::AbstractDomain) = domain.transforms
get_fwd(domain::AbstractDomain) = get_fwd(get_transform_plans(domain))
get_bwd(domain::AbstractDomain) = get_bwd(get_transform_plans(domain))
get_precision(domain::AbstractDomain) = domain.precision
memory_type(domain::AbstractDomain) = domain.MemoryType{domain.precision}

# Overloading
Base.size(domain::AbstractDomain) = (domain.Ny, domain.Nx)
Base.length(domain::AbstractDomain) = prod(size(domain))