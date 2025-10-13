# --------------------------------- Poisson bracket ----------------------------------------

# ----------------------------------- Construction -----------------------------------------

struct PoissonBracket <: NonLinearOperator
    diff_x::LinearOperator
    diff_y::LinearOperator
    quadratic_term::QuadraticTerm
    tmp1::AbstractArray
    tmp2::AbstractArray
    qt_left::AbstractArray
    qt_right::AbstractArray

    function PoissonBracket(ks, DomainType, MemoryType, precision, dealiased, real_transform, diff_x, diff_y, quadratic_term)

        # Allocate
        tmp1 = zeros(precision, length.(ks)) |> MemoryType
        tmp2 = similar(tmp1)
        qt_left = zeros(precision, length.(ks)) |> MemoryType
        qt_right = similar(qt_left)

        new(diff_x, diff_y, quadratic_term, tmp1, tmp2, qt_left, qt_right)
    end
end

operator_type(::Val{:poisson_bracket}, ::Type{_}) where {_} = PoissonBracket

function operator_args()
    @unpack MemoryType, precision, dealiased, real_transform = domain_kwargs
    return Ns, DomainType, MemoryType, precision, dealiased, real_transform
end

function operator_dependencies(::Val{T}, ::Type{PoissonBracket}) where {T}
    [OperatorRecipe(:diff_x), OperatorRecipe(:diff_y), OperatorRecipe(:quadratic_term)]
end

# Cache alternatives for poisson_bracket
# 1. vx, vy, tmp, qt_left, qt_right (vx and vy, can be reused in theory)
# 2. tmp1, tmp2, qt_left, qt_right (takes up less space)

#PoissonBracket(quadratic_term, diff_x, diff_y)

@inline function (op::PoissonBracket)(u_hat::AbstractArray, v_hat::AbstractArray)
    @unpack tmp1, tmp2, qt_left, qt_right, diff_x, diff_y, quadratic_term = op
    tmp1 = diff_x(u_hat)
    tmp2 = diff_y(v_hat)
    qt_left = quadratic_term(tmp1, tmp2)
    tmp1 = diff_x(v_hat)
    tmp2 = diff_y(u_hat)
    qt_right = quadratic_term(tmp1, tmp2)

    return qt_left .- qt_right
end

# function (op::PoissonBracket)(A::T, B::T) where {T<:AbstractArray}
#     @unpack tmp, vx, vy, qt, diff_x, diff_y = op
#     @unpack qt_left, qt_right = qt

#     # TODO perhpas check if needs to be updated?
#     if true
#         vx .= diff_y(B)
#         vy .= diff_x(B)
#     end

#     tmp .= diff_x(A)
#     quadratic_term(qt_left, tmp, vx)

#     tmp .= diff_y(A)
#     quadratic_term(qt_right, tmp, vy)

#     tmp .= qt_left - qt_right
# end

# #vx, vy, phi, out
# #poisson_bracket -> vx, vy (would be nice to keep in cache?)
# #Option to store velocity fields