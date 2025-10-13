# ------------------------------- Spectral Operators ---------------------------------------

# Abstract class that may inherit from a more abstract one
abstract type SpectralOperator end

# TODO check if needed
# For broadcasting the operators for composite operators
Base.broadcastable(op::SpectralOperator) = Ref(op)

# --------------------------------- Operator declaration -----------------------------------

struct OperatorRecipe{KwargsType<:NamedTuple}
    op::Symbol
    alias::Symbol
    kwargs::KwargsType

    function OperatorRecipe(op; alias=op, kwargs...)
        # TODO check that op is valid
        nt = NamedTuple(kwargs)
        new{typeof(nt)}(op, alias, nt)
    end
end

Base.hash(opdecl::OperatorRecipe, h::UInt) = hash((opdecl.op, opdecl.kwargs), h)
function Base.:(==)(od1::OperatorRecipe, od2::OperatorRecipe)
    od1.op == od2.op && od1.kwargs == od2.kwargs
end

# function Base.get(opdecl::OperatorRecipe, key, default)
#     hasproperty(opdecl, key) ? getproperty(opdecl, key) : default
# end

# ---------------------------------- Constructors ------------------------------------------

# TODO add @op for each operator
const DEFAULT_OPERATORS = [
    OperatorRecipe(:diff_x),
    OperatorRecipe(:diff_y),
    OperatorRecipe(:laplacian),
    OperatorRecipe(:solve_phi),
    OperatorRecipe(:poisson_bracket),
    OperatorRecipe(:quadratic_term)
]

# Catch all method, can be overwritten with specilization
operator_dependencies(::Val{_}, ::Type{__}) where {_,__} = ()

# TODO remove temporary method and implement OperatorRecipes properly
function prepare_operators(::Type{Domain}, operators::Vector, ks, Ns; domain_kwargs...)
    @unpack MemoryType, precision, dealiased, real_transform = domain_kwargs
    diff_y = ElwiseOperator(im .* ks[1])
    diff_x = ElwiseOperator(im .* transpose(ks[2]))
    laplacian = ElwiseOperator(get_laplacian(Domain, ks))
    quadratic_term = QuadraticTerm(Ns, Domain, MemoryType, precision, dealiased, real_transform)
    poisson_bracket = PoissonBracket(ks, Domain, MemoryType, precision, dealiased,
        real_transform, diff_x, diff_y, quadratic_term)
    solve_phi = SolvePhi(ks)

    spectral_operators = (diff_x=diff_x, diff_y=diff_y, laplacian=laplacian,
        quadratic_term=quadratic_term, poisson_bracket=poisson_bracket, solve_phi=solve_phi)

    return spectral_operators

end

# function prepare_operators(::Type{Domain}, operators::Vector, ks, Ns; domain_kwargs...)

#     cached = Dict{OperatorRecipe,SpectralOperator}()
#     aliases = 0 # TODO make set of aliases to throw error

#     function ensure!(declaration)
#         haskey(cached, declaration) && return cached[declaration]

#         # Extract fields from declaration
#         @unpack op, alias, kwargs = declaration

#         # Get construction information
#         OperatorType = operator_type(Val(op), Domain)
#         args = operator_args(Val(op), Domain, ks, Ns; domain_kwargs...)
#         # TODO throw Tuple warning or correct for non-tuple args

#         # TODO Something with aliases

#         # Get the dependencies based on the dependent declarations
#         dependencies = [ensure!(dep) for dep in operator_dependencies(Val(op), OperatorType)]

#         # Construct the operator
#         cached[declaration] = OperatorType(args..., dependencies...; kwargs...)
#     end

#     # TODO store aliases, to check if two same aliases, what to do in that case?

#     spectral_operators = []
#     for declaration in operators
#         push!(spectral_operators, declaration.alias => ensure!(declaration))
#     end

#     return NamedTuple(spectral_operators)

# end

# -------------------------------- Linear operators ----------------------------------------

# Abstract type that all Linear operators inherit from
abstract type LinearOperator{T} <: SpectralOperator end

# TODO add Algebraic rules of Linear operators
import Base: *, +, -, ^
*(a::Number, op::LinearOperator) = typeof(op)(a .* op.coeffs)

# Allows composite operators
# TODO add some sort of promotion rules
+(a::LinearOperator, b::LinearOperator) = typeof(op)(a.coeffs .+ b.coeffs)
-(a::LinearOperator, b::LinearOperator) = typeof(op)(a.coeffs .- b.coeffs)
^(op::LinearOperator, power::Number) = typeof(op)(op.coeffs .^ power)

# ----------------------------- Elementwise operator ---------------------------------------

struct ElwiseOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number

    function ElwiseOperator(coeffs; order=1)
        new{typeof(coeffs)}(coeffs .^ order, order)
    end
end

# Out-of-place operator
@inline (op::ElwiseOperator)(u::AbstractArray) = op.coeffs .* u

# To be able to use @. without applying LinearOperator to array element
import Base.Broadcast: broadcasted
function broadcasted(op::ElwiseOperator, x)
    return broadcasted(*, op.coeffs, x)
end

# -------------------------------- Matrix operator -----------------------------------------

struct MatrixOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number
end

# Out-of-place operator # TODO figure out what to do here
@inline (op::MatrixOperator)(u::AbstractArray) = op.coeffs * u

"""
    Laplacian = -kx' .^ 2 .- ky .^ 2
    invLaplacian = Laplacian .^ -1
    invLaplacian[1] = 0 # First entry will always be NaN or Inf
"""

include("spatialDerivatives.jl")

# --------------------------- Non-Linear operators -----------------------------------------

# Abstract type that all NonLinear operators inherit from
abstract type NonLinearOperator <: SpectralOperator end

include("quadraticTerm.jl")
include("poissonBracket.jl")

# -------------------------------------- Others --------------------------------------------

include("solvePhi.jl")
include("spectralFunctions.jl")
#include("sources.jl")


# ----------------------------- OperatorRecipe macro ----------------------------------

# TODO implement
macro op()
end




























## -------------------------------- TESTING BELOW ------------------------------------------

# using CUDA
# using FFTW
# kx, ky = prepare_frequencies(256, 256, 0.1, 0.1, true, Float64, true)

# diff_x = ElwiseOperator(transpose(im * kx))
# diff_y = ElwiseOperator(im * ky)

# u = rand(129, 256) |> CuArray

# (diff_x^2)(u)
# diff_y(u)


# # Prepare frequencies
# kx, ky = prepare_frequencies(Nx, Ny, dx, dy, use_cuda, precision, real_transform)

# # Prepare transform plans
# transforms = prepare_transform_plans(Nx, Ny, use_cuda, precision, real_transform)

# # TODO move to Domain
# # Prepare spectral operators
# operators = prepare_operators(operators, kx, ky, Nx, Ny, use_cuda=use_cuda,
#     precision=precision, real_transform=real_transform, dealiased=dealiased)

# # TODO make robust
# function construct_operators()#::Domain)
#     operators = (diff_x=ElwiseOperator(rand(256, 256)),
#         diff_y=ElwiseOperator(rand(256, 256)),
#         diff_xx=ElwiseOperator(rand(256, 256)),
#         diff_yy=ElwiseOperator(rand(256, 256)),
#         laplacian=ElwiseOperator(rand(256, 256)),
#         inv_laplacian=ElwiseOperator(rand(256, 256)),
#         hyper_laplacian=ElwiseOperator(rand(256, 256)))
# end







# domain = Domain(domain, add_operators=[], add_aliases=[])
# domain = Domain(256, 256, Lx=1, Ly=1, operators=:default, add_operators=[
#     @op ∂xx = diff_x(order=2),
#     @op hyper_laplacian = laplacian(order=3), 
#     @op spectral_exp, @op spectral_log,
#     @op solve_phi(boussinesq=true)
# ])

# @opdecl ∂xx = diff_x(order=2) => OperatorRecipe(:diff_x, order=2, alias=∂xx)

# Union{Symbol,Vector{OperatorRecipe}}

# function prepare_operators(::Type{Domain}, operators, kx, ky, Nx, Ny; MemoryType=MemoryType,
#     precision=precision, real_transform=real_transform, dealiased=dealiased)

#     cache = Dict{Symbol,SpectralOperator}()

#     # Figure out aliases
#     #[:∂x, :∂y, :laplacian, :poisson_bracket] -> [:diff_x, :diff_y, :laplacian, :poisson_bracket]

#     # Figure out dependencies
#     #[:∂x, :∂y, :laplacian, :poisson_bracket] -> [:diff_x, :diff_y, :quadratic_term]
#     #operator_dependencies()

#     # Combine the two sets
#     #[:diff_x, :diff_y, :laplacian, :poisson_bracket] U [:diff_x, :diff_y, :quadratic_term] = [:diff_x, :diff_y, :laplacian, :quadratic_term, :poisson_bracket]

#     # Construct the operators in the union based on order?
#     #[:diff_x, :diff_y, :laplacian, :quadratic_term, :quadratic_term] -> [ElwiseOperator, ElwiseOperator, ElwiseOperator, QuadraticTerm, PoissonBracket]

#     # Link operator to alias
#     #[:∂x, :∂y, :laplacian, :poisson_bracket], [:diff_x, :diff_y, :laplacian, :poisson_bracket], [:diff_x, :diff_y, :laplacian, :quadratic_term, :poisson_bracket], [ElwiseOperator, ElwiseOperator, ElwiseOperator, QuadraticTerm, PoissonBracket]
#     #-> spectral_operators = (:∂x=ElwiseOperator, :∂y=ElwiseOperator, :laplacian=ElwiseOperator, :poisson_bracket=PoissonBracket)

#     for op in operators
#         cache[op] = construct_operator(Val(op), Domain, (kx, ky), (Nx, Ny),
#             MemoryType=MemoryType, precision=precision, real_transform=real_transform,
#             dealiased=dealiased)
#     end

#     # Link operators to request
#     spectral_operators = (; cache...)#op = cache[op] for op in operators)

#     return spectral_operators

# end




















# using CUDA
# quadratic_term = QuadraticTerm((256, 256), Domain, CuArray, Float64, dealiased=true, real_transform=true)

# quadratic_term.transforms
# quadratic_term.dealiasing_coefficient
# quadratic_term.U
# quadratic_term.V
# quadratic_term.up
# quadratic_term.vp
# quadratic_term.padded

# using LinearAlgebra

# u = rfft(rand(256, 256)) |> CuArray
# v = rfft(rand(256, 256)) |> CuArray
# out = rfft(rand(256, 256)) |> CuArray
# quadratic_term(out, u, v)

# # Allocate extra arrays for in-place operations
# qt_left = zeros(complex(precision), length(ky), length(kx))
# use_cuda ? qt_left = adapt(CuArray, qt_left) : nothing
# qt_right = zero(qt_left)