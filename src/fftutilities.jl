using FFTW
using LinearAlgebra
export TransformPlans, FFTPlans, rFFTPlans, spectral_transform, spectral_transform!, 
multi_fft, multi_ifft,

# Collection of plans for a transform and its inverse

abstract type TransformPlans end

struct FFTPlans <: TransformPlans
    FT::FFTW.Plan
    iFT::FFTW.Plan
end

struct rFFTPlans <: TransformPlans
    FT::FFTW.Plan
    iFT::FFTW.Plan
end

# General transform plan
function spectral_transform(U::T, p::FFTW.Plan) where {T<:AbstractArray}
    mapslices(u -> p * u, U, dims=(1, 2))
end

function spectral_transform(U::T, p::FFTW.Plan) where {T<:Union{Tuple,Vector}}
    map(u -> p * u, U)
end

function spectral_transform!(du::DU, u::U, p::FFTW.Plan) where {DU<:AbstractArray,U<:AbstractArray}
    @assert ndims(du) == ndims(u)
    idx = ntuple(_ -> :, ndims(p))
    for i in axes(u, ndims(p)+1)
        mul!(du[idx..., i], p, u[idx..., i])
    end
end

function spectral_transform!(du::DU, u::U, p::FFTW.Plan) where {DU<:Union{Tuple,Vector}, 
    U<:Union{Tuple,Vector}}
    for i in eachindex(u)
        mul!(du[i], p, u[i])
    end
end

# ------------------------------------- Old ------------------------------------------------
# TODO make these typed or remove

# Fourier transform applied to "stacked" fields
function multi_fft(U::AbstractArray, plans::TransformPlans)
    mapslices(u -> plans.FT * u, U, dims=(1, 2))
end

function multi_fft(U::Union{Tuple,Vector}, plans::TransformPlans)
    map(u -> plans.FT * u, U)
end

# TODO understand why plans::TransformPlans does not work
# Inverse Fourier transform
function multi_ifft(U::AbstractArray, plans)
    mapslices(u -> plans.iFT * u, U, dims=(1, 2))
end

function multi_ifft(U::Union{Tuple,Vector}, plans::TransformPlans)
    map(u -> plans.iFT * u, U)
end