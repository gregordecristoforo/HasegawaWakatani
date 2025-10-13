# ------------------------------ Spectral functions ----------------------------------------

# ---------------------------------- Construction ------------------------------------------

struct SpectralFunction{F<:Function} <: SpectralOperator
    f::F
    out::AbstractArray # TODO specialize?
    #args, kwargs? TODO perhaps expand upon this
    quadratic_term::QuadraticTerm
end

operator_type(::Union{Val{:reciprocal},Val{:spectral_exp},Val{:spectral_expm1},
        Val{:spectral_log}}, ::Type{_}) where {_} = SpectralFunction

operator_dependencies(::Val{_}, ::Type{SpectralFunction}) where {_} = [OperatorRecipe(:quadratic_term)]

function operator_args(::Val{:reciprocal}, ::Type{_}, Ns, ks; domain_kwargs...) where {_}
    return u -> div(1, u), zeros(ks[2], ks[1]) # TODO fix a permanent way to construct these temp arrays
end

function operator_args(::Val{:spectral_exp}, ::Type{_}, Ns, ks; domain_kwargs...) where {_}
    return exp, zeros(ks[2], ks[1])
end

function operator_args(::Val{:spectral_expm1}, ::Type{_}, Ns, ks; domain_kwargs...) where {_}
    return expm1, zeros(ks[2], ks[1])
end

function operator_args(::Val{:spectral_log}, ::Type{_}, Ns, ks; domain_kwargs...) where {_}
    return log, zeros(ks[2], ks[1])
end

# ------------------------------------ Apply function --------------------------------------

@inline function (op::SpectralFunction)(u::AbstractArray)
    out = op.out
    @unpack U, V, up, padded, transforms, dealiasing_coefficient = op.quadratic_term
    mul!(U, bwd(transforms), padded ? pad!(up, u, typeof(transforms)) : u)
    V .= op.f.(dealiasing_coefficient * U)
    #print(V)
    mul!(padded ? up : out, fwd(transforms), V)
    padded ? unpad!(out, up, typeof(transforms)) / dealiasing_coefficient : up
end