# ------------------------------------------------------------------------------------------
#                                    Solve Phi Operator                                     
# ------------------------------------------------------------------------------------------

# TODO make mutable? Or rethink alltogether
struct SolvePhi{T<:AbstractArray} <: SpectralOperator
    laplacian_inv::T
    phi::T

    function SolvePhi(domain)
        laplacian = complex(get_laplacian(domain))
        laplacian_inv = laplacian .^ -1
        @allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf
        phi = similar(laplacian)
        new{typeof(phi)}(laplacian_inv, phi)
    end
end

function build_operator(::Val{:solve_phi}, domain::AbstractDomain; boussinesq=true,
                        kwargs...)
    _build_operator(Val(:solve_phi), domain, Val(boussinesq); kwargs...)
end

function _build_operator(::Val{:solve_phi}, domain::Domain, ::Val{true}; kwargs...)
    SolvePhi(domain)
end

# ------------------------------------- Main Methods ---------------------------------------

# TODO implement Non-Bousinesq method issue [#24](https://github.com/JohannesMorkrid/HasegawaWakatani.jl/issues/24)
function _build_operator(::Val{:solve_phi}, domain::Domain, ::Val{false}; kwargs...)
    SolvePhi(domain)
end

@inline (op::SolvePhi)(u_hat::AbstractArray) = op.phi .= op.laplacian_inv .* u_hat