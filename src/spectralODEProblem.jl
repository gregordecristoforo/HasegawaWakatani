using FFTW

# TODO add get_velocity=vExB,

mutable struct SpectralODEProblem{LType<:Function,NType<:Function,D<:Domain,u0Type<:AbstractArray,
    u0_hatType<:AbstractArray,tType<:AbstractArray,pType<:Dict,N<:Number,RM<:Function,kwargsType}

    L::LType
    N::NType
    domain::D
    u0::u0Type
    u0_hat::u0_hatType

    tspan::tType
    p::pType

    # Passed onto something # TODO find out what this something is
    dt::N
    remove_modes::RM
    kwargs::kwargsType

    function SpectralODEProblem(N::Function, domain::Domain, u0, tspan;
        p=Dict(), dt=0.01, kwargs...)

        # If no linear operator given, assume there is non
        function L(u, d, p, t)
            zero(u)
        end

        SpectralODEProblem(L, N, domain, u0, tspan, p=p, dt=dt, kwargs...)
    end

    # function SpectralODEProblem(L::F, N::F, domain::D, u0, tspan; p=Dict(),
    #     dt=0.01, inverse_transformation::F=identity) where {F<:Function, D}
    function SpectralODEProblem(L::Function, N::Function, domain::Domain, u0, tspan;
        p=Dict(), dt=0.01, remove_modes=remove_nothing, kwargs...)

        sz = size(domain.transform.iFT)
        allocation_size = (sz..., size(u0)[length(sz)+1:end]...)
        u0_hat = zeros(eltype(domain.transform.iFT), allocation_size...)
        spectral_transform!(u0_hat, u0, domain.transform.FT)
        remove_modes(u0_hat, domain)

        if length(tspan) != 2
            throw("tspan should have exactly two elements tsart and tend")
        end

        new{typeof(L),typeof(N),typeof(domain),typeof(u0),typeof(u0_hat),typeof(tspan),
            typeof(p),typeof(dt),typeof(remove_modes),typeof(kwargs)}(L, N, domain, u0, u0_hat,
            tspan, p, dt, remove_modes, kwargs)
    end
end

#Need to handle kwargs like dt = 0.01, inverse_transformation::F=identity somewhere!

function Base.display(prob::SpectralODEProblem)
    println(typeof(prob))
end