# TODO make mutable? Or rethink alltogether
struct SolvePhi{T<:AbstractArray} <: SpectralOperator
    laplacian_inv::T
    phi::T

    function SolvePhi(ks)
        laplacian = complex(get_laplacian(Domain, ks))
        laplacian_inv = laplacian .^ -1
        CUDA.@allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf
        phi = similar(laplacian)
        new{typeof(phi)}(laplacian_inv, phi)
    end
end

operator_type(::Val{:solve_phi}, ::Type{Domain}) = SolvePhi

function operator_args(::Val{:solve_phi}, ::Type{Domain}, ks, Ns; domain_kwargs...)
    laplacian = get_laplacian(Domain, ks)
    laplacian_inv = laplacian .^ -1
    CUDA.@allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf
    phi = similar(laplacian)
    return laplacian_inv, phi
end

@inline function (op::SolvePhi)(u_hat::AbstractArray)
    op.phi .= op.laplacian_inv .* u_hat
end