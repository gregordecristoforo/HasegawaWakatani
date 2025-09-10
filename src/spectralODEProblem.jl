
# Inspired by SciMLBase, singleton type
struct NullParameters end

# TODO add get_velocity=vExB,

mutable struct SpectralODEProblem{LType<:Function,NType<:Function,D<:AbstractDomain,u0Type<:AbstractArray,
    u0_hatType<:AbstractArray,tType,pType,N<:Number,RMType<:Function,kwargsType}

    L::LType
    N::NType
    domain::D
    u0::u0Type
    u0_hat::u0_hatType
    tspan::tType
    p::pType
    # TODO add in-place flag?

    dt::N # Passed onto something TODO find out what this something is
    remove_modes::RMType
    kwargs::kwargsType

    function SpectralODEProblem(N::Function, domain::AbstractDomain, u0, tspan;
        p=NullParameters(), dt=0.01, remove_modes::Function=remove_nothing, kwargs...)

        # If no linear operator given, assume there is non
        L(u, d, p, t) = zero(u)

        SpectralODEProblem(L, N, domain, u0, tspan, p=p, dt=dt, remove_modes=remove_modes, kwargs...)
    end

    function SpectralODEProblem(L::Function, N::Function, domain::AbstractDomain, u0, tspan;
        p=NullParameters(), dt=0.01, remove_modes::Function=remove_nothing, kwargs...)

        # Transform to CUDA
        domain.use_cuda ? u0 = CuArray(u0) : nothing # TODO option for 32 or 64 bit CUDA

        # Allocate array for spectral modes 
        sz = size(domain.transform.iFT)
        allocation_size = (sz..., size(u0)[length(sz)+1:end]...)
        u0_hat = zeros(eltype(domain.transform.iFT), allocation_size...)

        # Used for normal Fourier transform
        eltype(domain.transform.FT) <: Complex ? u0 = complex(u0) : nothing

        # Transform to CUDA
        domain.use_cuda ? u0_hat = CuArray(u0_hat) : nothing # This controls precision

        # Compute 
        spectral_transform!(u0_hat, u0, domain.transform.FT)
        remove_modes(u0_hat, domain)

        # Handle timespan
        length(tspan) != 2 ? throw("tspan should have exactly two elements") : nothing
        tspan = promote(first(tspan), last(tspan))

        #dt = Float32(dt)

        new{typeof(L),typeof(N),typeof(domain),typeof(u0),typeof(u0_hat),typeof(tspan),
            typeof(p),typeof(dt),typeof(remove_modes),typeof(kwargs)}(L, N, domain, u0, u0_hat,
            tspan, p, dt, remove_modes, kwargs)
    end
end

#Need to handle kwargs like dt = 0.01, inverse_transformation::F=identity somewhere!

function Base.show(io::IO, m::MIME"text/plain", prob::SpectralODEProblem)
    print(io, nameof(typeof(prob)), "(", nameof(prob.L), ",", nameof(prob.N), ";dt=", prob.dt)
    typeof(prob.p) == NullParameters ? nothing : print(io, ",p=", prob.p)
    println(io, "):")
    println(io, "remove_modes: ", nameof(prob.remove_modes))
    print(io, "domain: ")
    show(io, m, prob.domain)
    print(io, "\ntimespan: ")
    show(io, prob.tspan)
    print(io, "\nu0: ")
    show(io, m, prob.u0)
end