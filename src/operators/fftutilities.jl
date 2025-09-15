abstract type TransformPlans end

"""
    FFTPlans{FT<:FFTW.Plan, iFT<:FFTW.Plan} 
  Collection of transform plans using standard FFT.
"""
struct FFTPlans{FWD<:FFTW.Plan,BWD<:FFTW.Plan} <: TransformPlans
    FT::FWD
    iFT::BWD
end

"""
    rFFTPlans{FT<:FFTW.Plan, iFT<:FFTW.Plan}
  Collection of transform plans using real FFT (rFFT), utilizing hermitian symmetry.
"""
struct rFFTPlans{FWD<:FFTW.Plan,BWD<:FFTW.Plan} <: TransformPlans
    FT::FWD
    iFT::BWD
end

"""
    spectral_transform!(U<:AbstractArray, transformplan<:FFTW.Plan)
    spectral_transform!(U<:Union{Tuple,Vector}, transformplan<:FFTW.Plan)
  Spectral transform, applies transform plan p to u in-place returning du.
"""
function spectral_transform!(du, u, p::P) where {P<:FFTW.Plan}
    _spectral_transform!(du, u, p)
end

function _spectral_transform!(du, u::AbstractArray{<:Number}, p::P) where {P<:FFTW.Plan}
    idx = ntuple(_ -> :, ndims(p))
    for i in axes(u, ndims(p) + 1)
        mul!(view(du, idx..., i), p, u[idx..., i])
    end
end

function _spectral_transform!(du, u::AbstractArray{<:AbstractArray}, p::P) where {P<:FFTW.Plan}
    for i in eachindex(u)
        _spectral_transform!(du[i], p, u[i])
    end
end

using ComponentArrays
# TODO move this implementation in extensions
function _spectral_transform!(du, u::ComponentArray, p::P) where {P<:FFTW.Plan}
    for k in keys(u)
        _spectral_transform!(getproperty(du, k), getproperty(u, k), p)
    end
end

# TODO optimize spectral_transform methods, maybe use spectral_transform!
"""
    spectral_transform(U<:AbstractArray, transformplan<:FFTW.Plan)
    spectral_transform(U<:Union{Tuple,Vector}, transformplan<:FFTW.Plan)
  Spectral transform out-of-place, applies transform plan p to U. 
"""
function spectral_transform(U::T, p::P) where {T<:AbstractArray,P<:FFTW.Plan}
    mapslices(u -> p * u, U, dims=(1, 2))
end

function spectral_transform(U::T, p::P) where {T<:Union{Tuple,Vector},P<:FFTW.Plan}
    map(u -> p * u, U)
end

# function spectral_transform(U, p::P) where {P<:FFTW.Plan}
#     allocate_coefficients()
#     spectral_transform!
# end

"""
    Base.show(io::IO, transformplans::TransformPlans)

  Pretty-print `TransformPlans`.
"""
function Base.show(io::IO, transformplans::TransformPlans)
    typename = nameof(typeof(transformplans))

    fwd = getfield(transformplans, :FT)
    bwd = getfield(transformplans, :iFT)

    # TODO perhaps do a better check if the plan is real
    if transformplans isa rFFTPlans
        println(io, typename)
        println(io, "  forward (real→complex): ", fwd)
        println(io, "  backward (complex→real): ", bwd)
    else
        println(io, typename)
        println(io, "  forward: ", fwd)
        println(io, "  backward :", bwd)
    end
end

"""
    Base.show(io::IO, ::MIME"text/plain", transformplans::TransformPlans)

  Compact one-line show of TransformPlans for use in arrays and etc.
"""
function Base.show(io::IO, ::MIME"text/plain", transformplans::TransformPlans)
    typename = nameof(typeof(transformplans))
    fwd = getfield(transformplans, :FT)
    bwd = getfield(transformplans, :iFT)

    print(io, typename, "(fwd: ", fwd, ", bwd: ", bwd, ")")
end