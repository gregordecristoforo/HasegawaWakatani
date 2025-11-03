# ------------------------------------------------------------------------------------------
#                                    Spectral Operators                                     
# ------------------------------------------------------------------------------------------

# ---------------------------------------- General -----------------------------------------

# Abstract class that may inherit from a more abstract one
abstract type SpectralOperator end

# TODO check if needed
# For broadcasting the operators for composite operators
Base.broadcastable(op::SpectralOperator) = Ref(op)

# ------------------------------------ Operator Recipe -------------------------------------

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

Base.hash(oprecipe::OperatorRecipe, h::UInt) = hash((oprecipe.op, oprecipe.kwargs), h)
function Base.:(==)(or1::OperatorRecipe, or2::OperatorRecipe)
    or1.op == or2.op && or1.kwargs == or2.kwargs
end

# function Base.get(oprecipe::OperatorRecipe, key, default)
#     hasproperty(oprecipe, key) ? getproperty(oprecipe, key) : default
# end

# ------------------------------------- Constructors ---------------------------------------

# TODO add @op for each operator

# Catch all method, can be overwritten with specilization
operator_dependencies(::Val{_}, ::Type{__}) where {_,__} = ()

function get_operator_recipes(operators::Symbol)
    if operators == :default
        return [OperatorRecipe(:diff_x; order=1),
                OperatorRecipe(:diff_y; order=1),
                OperatorRecipe(:laplacian; order=1),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket)]
    elseif operators == :SOL
        return [OperatorRecipe(:diff_x),
                OperatorRecipe(:diff_y),
                OperatorRecipe(:laplacian),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket),
                OperatorRecipe(:quadratic_term)]
    elseif operators == :all
        return [OperatorRecipe(:diff_x; order=1),
                OperatorRecipe(:diff_y; order=1),
                OperatorRecipe(:laplacian; order=1),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket),
                OperatorRecipe(:quadratic_term),
                OperatorRecipe(:spectral_log),
                OperatorRecipe(:spectral_exp),
                OperatorRecipe(:spectral_expm1),
                OperatorRecipe(:reciprocal)]
    else
        error()
    end
end

"""
    prepare_operator_recipes(operators::Symbol, additional_operators::Vector{<:OperatorRecipe})

  Use switches to get a list of `OperatorRecipe`s and append the `additional_operators` to it.
"""
function prepare_operator_recipes(operators::Symbol,
                                  additional_operators::Vector{<:OperatorRecipe})
    # Determine operators trough switches
    recipes = get_operator_recipes(operators)

    # Combine with additional_operators 
    recipes = vcat(recipes, additional_operators)

    return recipes
end

# function add_aliases!(operators, aliases, cache)
#     for alias in aliases
#         # Get last, as thats whats being aliases
#         operator = last(alias)

#         cache[]
#     end
#     vcat()
# end

function build_operators(domain::AbstractDomain; operators::Symbol=:default,
                         aliases::Vector{Pair{Symbol,Symbol}}=Pair{Symbol,Symbol}[],
                         additional_operators::Vector{<:OperatorRecipe}=OperatorRecipe[],
                         problem_kwargs...)
    # Collects all recipes needed to be built
    recipes = prepare_operator_recipes(operators, additional_operators)

    # Used to only have to build each operator once
    cache = Dict{OperatorRecipe,SpectralOperator}()

    function ensure!(cache, recipe, domain, problem_kwargs)
        # To not construct the same operator twice
        if haskey(cache, recipe)
            return cache[recipe]
        end

        # Makes the code more readable
        operator = recipe.op

        # Get dependencies and construct recursively
        dependencies = [dependency.op => ensure!(cache, dependency, domain, problem_kwargs)
                        for dependency in operator_dependencies(Val(operator), Domain)]

        # Collect kwargs from recipe, problem and combine with dependency references
        kwargs = (; recipe.kwargs..., problem_kwargs..., dependencies...)

        # Build operator and add to cache
        cache[recipe] = build_operator(Val(operator), domain; kwargs...)
    end

    # Sort out aliases

    # Construct NamedTuple
    spectral_operators = (;
                          [recipe.op => ensure!(cache, recipe, domain, problem_kwargs)
                           for recipe in recipes]...)

    return spectral_operators
end

# ----------------------------------- Linear Operators -------------------------------------

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

# --------------------------------- Elementwise Operator -----------------------------------

struct ElwiseOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number

    ElwiseOperator(coeffs; order=1) = new{typeof(coeffs)}(coeffs .^ order, order)
end

# Out-of-place operator
@views @inline (op::ElwiseOperator)(u::AbstractArray) = op.coeffs .* u

# To be able to use @. without applying LinearOperator to array element
import Base.Broadcast: broadcasted
broadcasted(op::ElwiseOperator, x) = broadcasted(*, op.coeffs, x)

# ------------------------------------ Matrix Operator -------------------------------------

struct MatrixOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number
end

# Out-of-place operator # TODO figure out what to do here
@views @inline (op::MatrixOperator)(u::AbstractArray) = op.coeffs * u

include("spatialDerivatives.jl")

# --------------------------------- Non-Linear Operators -----------------------------------

# Abstract type that all NonLinear operators inherit from
abstract type NonLinearOperator <: SpectralOperator end

include("quadraticTerm.jl")
include("poissonBracket.jl")

# ---------------------------------------- Others ------------------------------------------

include("solvePhi.jl")
include("spectralFunctions.jl")
#include("sources.jl")

# --------------------------------- OperatorRecipe Macro -----------------------------------

# TODO implement
macro op()
end

## -------------------------------- TESTING BELOW ------------------------------------------

# @op ∂xx = diff_x(order=2) => OperatorRecipe(:diff_x, order=2, alias=∂xx)

# function prepare_operators(::Type{Domain}, operators, kx, ky, Nx, Ny; MemoryType=MemoryType,
#     precision=precision, real_transform=real_transform, dealiased=dealiased)

#     cache = Dict{Symbol,SpectralOperator}()

#     # Figure out aliases
#     #[:∂x, :∂y, :laplacian, :poisson_bracket] -> [:diff_x, :diff_y, :laplacian, :poisson_bracket]

#     # Link operator to alias
#     #[:∂x, :∂y, :laplacian, :poisson_bracket], [:diff_x, :diff_y, :laplacian, :poisson_bracket], [:diff_x, :diff_y, :laplacian, :quadratic_term, :poisson_bracket], [ElwiseOperator, ElwiseOperator, ElwiseOperator, QuadraticTerm, PoissonBracket]
#     #-> spectral_operators = (:∂x=ElwiseOperator, :∂y=ElwiseOperator, :laplacian=ElwiseOperator, :poisson_bracket=PoissonBracket)

# end

# diff_x = ∂x = Dx
# diff_y = ∂y  = Dy
# diff_xx = ∂xx = Dxx = ∂x² = (∂x^2)
# diff_yy = ∂yy = Dyy = ∂y² (∂y^2)
# diff_xn = ∂xn = Dxn (∂x^n)
# diff_yn = ∂yn = Dyn (∂y^n)
# laplacian = diffusion = Δ
# hyper_laplacian = hyper_diffusion (Δ^p)