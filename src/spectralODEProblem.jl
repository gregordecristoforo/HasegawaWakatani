
# Inspired by SciMLBase, singleton type
"""
    Nullparamters()
  Singleton to signalize that the user has not specified any parameters.
"""
struct NullParameters end

# TODO add get_velocity=vExB,

"""
    SpectralODEProblem(L::Function, N::Function, domain::AbstractDomain, u0, tspan;
        p=NullParameters(), dt=0.01, remove_modes::Function=remove_nothing, kwargs...)
    SpectralODEProblem(N::Function, domain::AbstractDomain, u0, tspan;
        p=NullParameters(), dt=0.01, remove_modes::Function=remove_nothing, kwargs...)
  
  Collection of data needed to specify the spectral ODE problem to be solved. The user needs
   to specify the non-linear operator `N`, with the linear operator `L` being optional and 
   otherwise assumed tp be zero. In addition the `domain`, initial condition `u0` and
   timespan `tspan` needs to be specified. The parameters `p` will be passed onto the 
   RHS/operators and the timestep `dt` is used by the temporal scheme. There is also the 
   option to add a method `remove_modes` to remove certain modes after each timestep. Other
   `kwargs` can be stored in the struct, however these are currently unused.
"""
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

        # Prepare data structures
        u0 = prepare_initial_state(u0, domain)
        u0_hat = prepare_spectral_coefficients(u0, domain)

        # Remove unwanted modes
        remove_modes(u0_hat, domain)

        # Handle timespan
        length(tspan) != 2 ? throw("tspan should have exactly two elements") : nothing
        tspan = promote(first(tspan), last(tspan))

        #dt = convert(precision, dt)

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

function prepare_initial_state(u0, domain::Domain)
    # Transform to CUDA if used
    domain.use_cuda ? u0 = adapt(CuArray{domain.precision}, u0) : nothing

    # Used for normal Fourier transform
    eltype(domain.transform.FT) <: Complex ? u0 = complex(u0) : nothing

    return u0
end

function prepare_spectral_coefficients(u0, domain::Domain)
    # Allocate data structure for spectral modes
    u0_hat = allocate_coefficients(u0, domain)

    # Compute spectral initial conditions
    spectral_transform!(u0_hat, u0, get_fwd_transform(domain))

    return u0_hat
end

"""
    allocate_coefficients(u0, transformplans::TransformPlans)
  Recursively iterates trough the initial condition data structure to try to get to the 
  lowest level Array and then allocates the needed shape after applying the fwd transform.
"""
function allocate_coefficients(u0, domain::Domain)
    _allocate_coefficients(u0, domain)
end

# TODO perhaps clean up this logic
function _allocate_coefficients(u0::AbstractArray{<:Number}, domain::Domain)
    # Allocate array for spectral modes 
    sz = size(get_bwd(domain))
    allocation_size = (sz..., size(u0)[length(sz)+1:end]...)
    u0_hat = zeros(eltype(get_bwd(domain)), allocation_size)

    # Transform to CUDA
    domain.use_cuda ? adapt(CuArray, u0_hat) : u0_hat
end

function _allocate_coefficients(u0::AbstractArray{<:AbstractArray}, domain::Domain)
    [_allocate_coefficients(u, domain) for u in u0]
end