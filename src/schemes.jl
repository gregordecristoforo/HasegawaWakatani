# TODO remove, used for testing
#using Caching, InteractiveUtils
#@cache 

using MuladdMacro

# TODO inherit type from SciML 
abstract type AbstractODEAlgorithm end

# Equivalent to OrdinaryDiffEqConstantCache in SciML
abstract type AbstractTableau end

export MMS1, MSS2, MSS3, perform_step!, get_cache

# ----------------------------------------- MSS1 -------------------------------------------

mutable struct MSS1Cache
    #Coefficents are all 1
    u::AbstractArray
    c::AbstractArray
    dt::Number
end

struct MSS1 <: AbstractODEAlgorithm end

function get_cache(prob::SpectralODEProblem, alg::MSS1)
    kappa = prob.domain.SC.Laplacian
    nu = prob.p["nu"]
    dt = prob.dt
    u = prob.u0_hat
    c = @. (1 - nu * kappa * dt)^-1
    MSS1Cache(u, c, dt)
end

function unpack_cache(cache::MSS1Cache)
    cache.u, cache.c, cache.dt
end

@muladd function perform_step!(cache::MSS1Cache, prob::SpectralODEProblem, t::Number)
    u, c, dt = unpack_cache(cache)
    d, f, p = prob.domain, prob.f, prob.p

    # Perform step
    cache.u = c .* (u .+ dt * f(u, d, p, t))
end

# ----------------------------------------- MSS2 -------------------------------------------

struct MSS2 <: AbstractODEAlgorithm end

mutable struct MSS2Cache
    #Coefficents
    u::AbstractArray
    c::AbstractArray
    u0::AbstractArray
    u1::AbstractArray
    k0::AbstractArray
    tab::AbstractTableau
    step::Integer
end

# Coefficents/tableu
struct MSS2Tableau <: AbstractTableau
    g0::Number # = 3 / 2
    a0::Number # = -1 / 2
    a1::Number # = 2
    b0::Number # = -1
    b1::Number # = 2
end

function MSS2Tableau()
    g0 = 3 / 2
    a0 = -1 / 2
    a1 = 2
    b0 = -1
    b1 = 2
    MSS2Tableau(g0, a0, a1, b0, b1)
end

function get_cache(prob::SpectralODEProblem, alg::MSS2)
    tab = MSS2Tableau()
    kappa = prob.domain.SC.Laplacian
    nu = prob.p["nu"]
    dt = prob.dt
    u = prob.u0_hat
    c = @. (tab.g0 - nu * kappa * dt)^-1
    u0 = prob.u0_hat
    u1 = zeros(size(u))
    k0 = zeros(size(u))
    step = 1
    MSS2Cache(u, c, u0, u1, k0, tab, step)
end

function unpack_cache(cache::MSS2Cache)
    cache.u, cache.c, cache.u0, cache.u1, cache.k0, cache.tab, cache.step
end

@muladd function perform_step!(cache::MSS2Cache, prob::SpectralODEProblem, t::Number)
    u, c, u0, u1, k0, tab, step = unpack_cache(cache)
    a0, a1, b0, b1 = tab.a0, tab.a1, tab.b0, tab.b1
    d, f, p, dt = prob.domain, prob.f, prob.p, prob.dt

    if cache.step == 1
        cache.step += 1
        # Perform step using MSS1
        cache1 = get_cache(prob, MSS1())
        perform_step!(cache1, prob, t)
        cache.k0 = f(u0, d, p, t)
        cache.u1 = cache1.u
    else
        k1 = f(u1, d, p, t)
        # Step
        cache.u = c .* (a0 * u0 .+ a1 * u1 .+ dt * (b0 * k0 .+ b1 * k1))
        # Shifting values downwards    
        cache.k0 = k1
        cache.u0 = u1
        cache.u1 = cache.u
    end
end

# ----------------------------------------- MSS3 -------------------------------------------

struct MSS3 <: AbstractODEAlgorithm end

mutable struct MSS3Cache
    #Coefficents
    u::AbstractArray
    c::AbstractArray
    u0::AbstractArray
    u1::AbstractArray
    u2::AbstractArray
    k0::AbstractArray
    k1::AbstractArray
    tab::AbstractTableau
    step::Integer
end

# Coefficents/tableu
struct MSS3Tableau <: AbstractTableau
    g0::Number # = 3 / 2
    a0::Number # = -1 / 2
    a1::Number # = 2
    a2::Number # = 2
    b0::Number # = -1
    b1::Number # = 2
    b2::Number # = 2
end

function MSS3Tableau()
    g0 = 11 / 6
    a0 = 1 / 3
    a1 = -3 / 2
    a2 = 3
    b0 = 1
    b1 = -3
    b2 = 3
    MSS3Tableau(g0, a0, a1, a2, b0, b1, b2)
end

function get_cache(prob::SpectralODEProblem, alg::MSS3)
    tab = MSS3Tableau()
    kappa = prob.domain.SC.Laplacian
    nu = prob.p["nu"]
    dt = prob.dt
    u = prob.u0_hat
    c = @. (tab.g0 - nu * kappa * dt)^-1
    u0 = prob.u0_hat
    u1 = zeros(size(u))
    u2 = zeros(size(u))
    k0 = zeros(size(u))
    k1 = zeros(size(u))
    step = 1
    MSS3Cache(u, c, u0, u1, u2, k0, k1, tab, step)
end

function unpack_cache(cache::MSS3Cache)
    cache.u, cache.c, cache.u0, cache.u1, cache.u2, cache.k0, cache.k1, cache.tab, cache.step
end

@muladd function perform_step!(cache::MSS3Cache, prob::SpectralODEProblem, t::Number)
    u, c, u0, u1, u2, k0, k1, tab, step = unpack_cache(cache)
    a0, a1, a2, b0, b1, b2 = tab.a0, tab.a1, tab.a2, tab.b0, tab.b1, tab.b2
    d, f, p, dt = prob.domain, prob.f, prob.p, prob.dt

    if cache.step == 1
        cache.step += 1
        # Perform step using MSS1
        cache1 = get_cache(prob, MSS1())
        perform_step!(cache1, prob, t)
        cache.k0 = f(u0, d, p, t)
        cache.u1 = cache1.u
    elseif cache.step == 2
        cache.step += 1
        # Set up cache for stepping with MSS2
        cache2 = get_cache(prob, MSS2())
        cache2.step = 2
        cache2.k0 = cache.k0
        cache2.u1 = cache.u1
        # Perform step using MSS2
        perform_step!(cache2, prob, t)
        cache.k1 = cache2.k0
        cache.u2 = cache2.u
    else
        k2 = f(u2, d, p, t)
        # Step
        cache.u = c .* (a0 * u0 .+ a1 * u1 .+ a2 * u2 .+ dt * (b0 * k0 .+ b1 * k1 .+ b2 * k2))
        # Shifting values downwards    
        cache.k1 = k2
        cache.k0 = k1
        cache.u1 = u2
        cache.u0 = u1
        cache.u2 = cache.u
    end
end