# ----------------------------------- Spatial Derivatives ----------------------------------

# ---------------------------------- Construction interface --------------------------------

"""
    construct_operator(Val{op}, DomainType, args; kwargs...)

    ks = (kx, ky, kz, ...)

    domain = Domain(args; operators=[:diff_x, ...], kwargs...)
    diff_x = domain.operators.diff_x
"""

# ------------------------------------- Fourier Domain -------------------------------------

operator_type(::Union{Val{:diff_x},Val{:diff_y},Val{:laplacian}}, ::Type{_}) where {_} = ElwiseOperator

operator_args(::Val{:diff_x}, ::Type{Domain}, ks, Ns; domain_kwargs...) = (im .* transpose(ks[2]),)

operator_args(::Val{:diff_y}, ::Type{Domain}, ks, Ns; domain_kwargs...) = (im .* ks[1],)

# Helper function 
function get_laplacian(::Type{Domain}, ks)
    N = length(ks)
    reshaped = ntuple(i -> reshape(ks[i], ntuple(j -> j == i ? length(ks[i]) : 1, N)), N)
    return -mapreduce(k -> k .^ 2, (a, b) -> a .+ b, reshaped)
end

operator_args(::Val{:laplacian}, ::Type{Domain}, ks, Ns; domain_kwargs...) = (get_laplacian(Domain, ks),)

# 
# function construct_operator(::Val{:hyper_laplacian}, ::Type{Domain}, ks)
#     ElwiseOperator(get_laplacian(Domain, ks) .^ 3) # TODO possible to remove magic number?
# end

# # TODO check if operators like laplacians and diff_xx and diff_yy should be Complex 

# function construct_operator(::Val{:diff_xx}, ::Type{Domain}, ks)
#     ElwiseOperator(-transpose(ks[1]) .^ 2)
# end

# function construct_operator(::Val{:diff_yy}, ::Type{Domain}, ks)
#     ElwiseOperator(-ks[2] .^ 2)
# end

# ------------------------------- Chebyshev Domain -----------------------------------------

# TODO remove?

# Chebyshev
function chebyshev_differentiation_matrix(x::AbstractVector)
    Nx = length(x)
    D = zeros(Nx, Nx)
    for i in eachindex(x)
        for j in eachindex(x)
            if i == j
                D[j, i] = -x[j] / (2(1 - x[j]^2))
            else
                ci = i != 1 && i != Nx ? 1 : 2
                cj = j != 1 && j != Nx ? 1 : 2
                D[i, j] = (ci / cj) * (-1)^(i + j) / (x[i] - x[j])
            end
        end
    end
    D[1, 1] = (2 * (Nx - 1)^2 + 1) / 6
    D[Nx, Nx] = -(2 * (Nx - 1)^2 + 1) / 6

    return D
end

#x = cos.(π * (0:N-1) / (N - 1))
#D = chebyshev_differentiation_matrix(x)
#diff_x = MatrixOperator(chebyshev_differentiation_matrix(x))

#diff_xx(u) = stack(map(diff_x, eachslice(u, dims=2)))


# --------------------------------------- Aliases ------------------------------------------

# diff_x = ∂x = Dx
# diff_y = ∂y  = Dy
# diff_xx = ∂xx = Dxx = ∂x² = (∂x^2)
# diff_yy = ∂yy = Dyy = ∂y² (∂y^2)
# diff_xn = ∂xn = Dxn (∂x^n)
# diff_yn = ∂yn = Dyn (∂y^n)
# laplacian = diffusion = Δ
# hyper_laplacian = hyper_diffusion (Δ^p)