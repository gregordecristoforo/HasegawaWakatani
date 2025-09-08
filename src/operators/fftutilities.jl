abstract type TransformPlans end

struct FFTPlans{F,B} <: TransformPlans where {F<:FFTW.Plan,B<:FFTW.Plan}
    FT::F
    iFT::B
end

struct rFFTPlans{F,B} <: TransformPlans where {F<:FFTW.Plan,B<:FFTW.Plan}
    FT::F
    iFT::B
end

# General transform plans
function spectral_transform(U::T, p::P) where {T<:AbstractArray,P<:FFTW.Plan}
    mapslices(u -> p * u, U, dims=(1, 2))
end

function spectral_transform(U::T, p::P) where {T<:Union{Tuple,Vector},P<:FFTW.Plan}
    map(u -> p * u, U)
end

function spectral_transform!(du::DU, u::U, p::P) where {DU<:AbstractArray,U<:AbstractArray,P<:FFTW.Plan}
    @assert ndims(du) == ndims(u)
    idx = ntuple(_ -> :, ndims(p))
    for i in axes(u, ndims(p) + 1)
        mul!(view(du, idx..., i), p, u[idx..., i])
    end
end

function spectral_transform!(du::DU, u::U, p::P) where {DU<:Union{Tuple,Vector},
    U<:Union{Tuple,Vector},P<:FFTW.Plan}
    for i in eachindex(u)
        mul!(du[i], p, u[i])
    end
end

# ------------------------------------- Old ------------------------------------------------
# TODO make these typed or remove
# Collection of plans for a transform and its inverse

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