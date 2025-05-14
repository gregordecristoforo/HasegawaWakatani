using MuladdMacro
using UnPack

# TODO inherit type from SciML 
abstract type AbstractODEAlgorithm end

# Equivalent to OrdinaryDiffEqConstantCache in SciML
abstract type AbstractTableau end

# Equivalent to DECache in SciMLBase 
abstract type AbstractCache end

export MMS1, MSS2, MSS3, perform_step!, get_cache, unpack_cache

# ----------------------------------------- MSS1 -------------------------------------------

mutable struct MSS1Cache{U,C} <: AbstractCache
    #Coefficents are all 1
    u::U
    c::C
end

struct MSS1 <: AbstractODEAlgorithm end

function get_cache(prob::SpectralODEProblem, alg::MSS1)
    dt = prob.dt
    u = copy(prob.u0_hat)
    # Calculate linear differential operator coefficent once to cache it
    if CUDA.functional()
        D = prob.L(CUDA.ones(Float64, size(prob.domain.transform.iFT)), prob.domain, prob.p, 0)
    else
        D = prob.L(ones(size(prob.domain.transform.iFT)), prob.domain, prob.p, 0)
    end
    c = @. (1 - D * dt)^-1
    MSS1Cache(u, c)
end

function unpack_cache(cache::MSS1Cache)
    cache.u, cache.c
end

@muladd function perform_step!(cache::MSS1Cache, prob::SpectralODEProblem, t::Number)
    @unpack u, c = cache
    d, f, p, dt = prob.domain, prob.N, prob.p, prob.dt

    # Perform step
    cache.u .= c .* (u .+ dt .* f(u, d, p, t))
end

# ----------------------------------------- MSS2 -------------------------------------------

struct MSS2 <: AbstractODEAlgorithm end

# Coefficents/tableu
struct MSS2Tableau{T} <: AbstractTableau
    g0::T # = 3 / 2
    a0::T # = -1 / 2
    a1::T # = 2
    b0::T # = -1
    b1::T # = 2
end

function MSS2Tableau()
    g0 = 3 / 2
    a0 = -1 / 2
    a1 = 2.0
    b0 = -1.0
    b1 = 2.0
    MSS2Tableau(g0, a0, a1, b0, b1)
end

mutable struct MSS2Cache{U,C,K} <: AbstractCache
    #Coefficents
    u::U
    c::C
    u0::U
    u1::U
    k0::K
    tab::MSS2Tableau
    step::Int
end

function get_cache(prob::SpectralODEProblem, alg::MSS2)
    tab = MSS2Tableau()
    dt = prob.dt
    u = copy(prob.u0_hat)
    if CUDA.functional()
        D = prob.L(CUDA.ones(Float64, size(prob.domain.transform.iFT)), prob.domain, prob.p, 0)
    else
        D = prob.L(ones(size(prob.domain.transform.iFT)), prob.domain, prob.p, 0)
    end
    c = @. (3 / 2 - D * dt)^-1
    u0 = copy(prob.u0_hat)
    u1 = zero(u)
    k0 = zero(u)
    step = 1
    MSS2Cache(u, c, u0, u1, k0, tab, step)
end

function unpack_cache(cache::MSS2Cache)
    cache.u, cache.c, cache.u0, cache.u1, cache.k0, cache.tab, cache.step
end

@muladd function perform_step!(cache::MSS2Cache, prob::SpectralODEProblem, t::Number)
    @unpack u, c, u0, u1, k0, tab, step = cache
    @unpack a0, a1, b0, b1 = tab
    d, f, p, dt = prob.domain, prob.N, prob.p, prob.dt

    if cache.step == 1
        cache.step += 1
        # Perform step using MSS1
        cache1 = get_cache(prob, MSS1())
        perform_step!(cache1, prob, t)
        cache.k0 .= f(u0, d, p, t)
        cache.u1 .= cache1.u
        cache.u .= cache.u1
    else
        k1 = f(u1, d, p, t)
        # Step
        @. cache.u = c * (a0 * u0 + a1 * u1 + dt * (b0 * k0 + b1 * k1))
        # Shifting values downwards    
        cache.k0 .= k1
        cache.u0 .= u1
        cache.u1 .= cache.u
    end
end

# ----------------------------------------- MSS3 -------------------------------------------

struct MSS3 <: AbstractODEAlgorithm end

# Coefficents/tableu
struct MSS3Tableau{T} <: AbstractTableau
    g0::T # = 11 / 6
    a0::T # = 1 / 3
    a1::T # = -3/2
    a2::T # = 3
    b0::T # = 1
    b1::T # = -3
    b2::T # = 3
end

function MSS3Tableau()
    g0 = 11 / 6
    a0 = 1 / 3
    a1 = -3 / 2
    a2 = 3.0
    b0 = 1.0
    b1 = -3.0
    b2 = 3.0
    MSS3Tableau(g0, a0, a1, a2, b0, b1, b2) #This controls precision
end

mutable struct MSS3Cache{U,C,K} <: AbstractCache
    #Coefficents
    u::U
    c::C
    u0::U
    u1::U
    u2::U
    k0::K
    k1::K
    tab::MSS3Tableau
    step::Int
end

function get_cache(prob::SpectralODEProblem, alg::MSS3)
    tab = MSS3Tableau()
    dt = prob.dt
    u = copy(prob.u0_hat)
    if CUDA.functional()
        D = prob.L(CUDA.ones(Float64, size(prob.domain.transform.iFT)), prob.domain, prob.p, 0) #This controls precision
    else
        D = prob.L(ones(size(prob.domain.transform.iFT)), prob.domain, prob.p, 0)
    end
    c = @. (tab.g0 - D * dt)^-1
    u0 = copy(prob.u0_hat)
    u1 = zero(u)
    k0 = zero(u)
    u2 = zero(u)
    k1 = zero(u)
    step = 1
    MSS3Cache(u, c, u0, u1, u2, k0, k1, tab, step)
end

function unpack_cache(cache::MSS3Cache)
    cache.u, cache.c, cache.u0, cache.u1, cache.u2, cache.k0, cache.k1, cache.tab, cache.step
end

@muladd function perform_step!(cache::MSS3Cache, prob::SpectralODEProblem, t::Number)
    @unpack u, c, u0, u1, u2, k0, k1, tab, step = cache
    @unpack a0, a1, a2, b0, b1, b2 = tab
    d, f, p, dt = prob.domain, prob.N, prob.p, prob.dt

    if cache.step == 1
        cache.step += 1
        # Perform step using MSS1
        cache1 = get_cache(prob, MSS1())
        perform_step!(cache1, prob, t)
        cache.k0 .= f(u0, d, p, t)
        cache.u1 .= cache1.u
        #cache.u1 .= u0.*exp.(prob.L(ones(size(prob.domain.transform.iFT)), d, p, 0)*dt)

        # Perform 10000 steps with MSS1
        # N = 10000
        # cprob = deepcopy(prob)
        # cprob.dt = prob.dt / N
        # cache1 = get_cache(cprob, MSS1())
        # for n in 1:N
        #     perform_step!(cache1, cprob, t + (n - 1) * cprob.dt)
        # end
        # cache.u1 .= cache1.u

        cache.u .= cache.u1 # For output handling
    elseif cache.step == 2
        cache.step += 1
        # Set up cache for stepping with MSS2
        cache2 = get_cache(prob, MSS2())
        cache2.step = 2
        cache2.k0 .= cache.k0
        cache2.u1 .= cache.u1
        # Perform step using MSS2
        perform_step!(cache2, prob, t)
        cache.k1 .= cache2.k0
        cache.u2 .= cache2.u
        cache.u .= cache.u2 # For output handling
    else
        k2 = f(u2, d, p, t)
        # Step
        @. cache.u = c * (a0 * u0 + a1 * u1 + a2 * u2 + dt * (b0 * k0 + b1 * k1 + b2 * k2))
        # Shifting values downwards    
        cache.k0 .= k1
        cache.k1 .= k2
        cache.u0 .= u1
        cache.u1 .= u2
        cache.u2 .= cache.u
    end
end

# TODO investigate if possible to remove u1 for MSS2 and similarly u2 for MSS3
# This is possible, but makes the code less comprehensible