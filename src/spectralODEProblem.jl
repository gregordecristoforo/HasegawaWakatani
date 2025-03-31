using FFTW
export SpectralODEProblem

mutable struct SpectralODEProblem{LType<:Function,NType<:Function,D<:Domain,u0Type<:AbstractArray,
    u0_hatType<:AbstractArray,tType<:AbstractArray,pType<:Dict,N<:Number,kwargsType}

    L::LType
    N::NType
    domain::D
    u0::u0Type
    u0_hat::u0_hatType

    tspan::tType
    p::pType

    # Passed onto something # TODO find out what this something is
    dt::N
    kwargs::kwargsType
    #recover_fields!::Ltype

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
        p=Dict(), dt=0.01, kwargs...)

        u0_hat = spectral_transform(u0, domain.transform.FT)
        remove_zonal_modes!(u0_hat)

        if length(tspan) != 2
            throw("tspan should have exactly two elements tsart and tend")
        end

        new{typeof(L),typeof(N),typeof(domain),typeof(u0),typeof(u0_hat),typeof(tspan),
        typeof(p),typeof(dt),typeof(kwargs)}(L, N, domain, u0, u0_hat, tspan, p, dt, kwargs)
    end
end

#Need to handle kwargs like dt = 0.01, inverse_transformation::F=identity somewhere!

function Base.display(prob::SpectralODEProblem)
    println(typeof(prob))
end

#prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=0.1)