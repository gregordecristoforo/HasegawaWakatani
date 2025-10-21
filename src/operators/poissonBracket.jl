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

    function PoissonBracket(domain::AbstractDomain, diff_x::LinearOperator,
                            diff_y::LinearOperator, quadratic_term::QuadraticTerm)

        # Allocate
        tmp1 = zeros(spectral_size(domain)) |> memory_type(domain)
        tmp2 = similar(tmp1)
        qt_left = zeros(spectral_size(domain)) |> memory_type(domain)
        qt_right = similar(qt_left)

        new(diff_x, diff_y, quadratic_term, tmp1, tmp2, qt_left, qt_right)
    end
end

function operator_dependencies(::Val{:poisson_bracket}, ::Type{_}) where {_}
    [OperatorRecipe(:diff_x), OperatorRecipe(:diff_y), OperatorRecipe(:quadratic_term)]
end

function build_operator(::Val{:poisson_bracket}, domain::Domain; diff_x, diff_y,
                        quadratic_term, kwargs...)
    PoissonBracket(domain, diff_x, diff_y, quadratic_term)
end

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

# Cache alternatives for poisson_bracket
# 1. vx, vy, tmp, qt_left, qt_right (vx and vy, can be reused in theory)
# 2. tmp1, tmp2, qt_left, qt_right (takes up less space)

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