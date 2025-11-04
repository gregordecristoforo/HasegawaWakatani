# ------------------------------------------------------------------------------------------
#                                        Diagnostics                                        
# ------------------------------------------------------------------------------------------

# ---------------------------------- Main Functionality ------------------------------------

struct Diagnostic{N<:AbstractString,M<:Function,L<:AbstractString,A<:Tuple,
                  K<:NamedTuple}
    name::N
    method::M
    metadata::L
    assumes_spectral_state::Bool
    stores_data::Bool
    args::A
    kwargs::K

    function Diagnostic(; name::AbstractString,
                        method::Function,
                        metadata::String="",
                        assumes_spectral_state::Bool=false,
                        stores_data::Bool=true,
                        args::Tuple=(),
                        kwargs=NamedTuple())
        new{typeof(name),typeof(method),typeof(metadata),typeof(args),
            typeof(kwargs)}(name, method, metadata, assumes_spectral_state, stores_data,
                            args, kwargs)
    end
end

"""
    (diagnostic::Diagnostic)(state, prob, time)

  Apply diagnostic `method` with the right `args` and `kwargs` stored in the `Diagnostic`.
"""
@inline function (diagnostic::Diagnostic)(state, prob, time)
    diagnostic.method(state, prob, time, diagnostic.args...; diagnostic.kwargs...)
end

# Take diagnostic of initial field (id = initial diagnostic)
# if diagnostic.assumes_spectral_state
#   id = diagnostic.method(prob.u0_hat, prob, first(prob.tspan), diagnostic.args...;
#                          diagnostic.kwargs...)
#   else
#       id = diagnostic.method(u0, prob, first(prob.tspan), diagnostic.args...;
#                              diagnostic.kwargs...)
#   end

# -------------------------------- Building Of Diagnostics ---------------------------------

function build_diagnostic(method::Function; kwargs...)
    build_diagnostic(Val(Symbol(method)); kwargs...)
end

# function perform_diagnostic!(diagnostic::D, step::Integer, u::U, prob::SOP, t::N;
#                              store_hdf::Bool=true,
#                              store_locally::Bool=true) where {D<:Diagnostic,
#                                                               U<:AbstractArray,
#                                                               SOP<:SpectralODEProblem,
#                                                               N<:Number}
#     # u might be real or complex depending on previous handle_output and diagnostic.assumes_spectral_state

#     # Perform diagnostic # TODO make diagnostic.args..., u, prob, t
#     data = diagnostic.method(u, prob, t, diagnostic.args...; diagnostic.kwargs...)

#     if !isnothing(data)
#         # Calculate index
#         idx = step ÷ diagnostic.sample_step + 1

#         store_hdf ? write_data(diagnostic, idx, data, t) : nothing

#         store_locally ? write_local_data(diagnostic, idx, data, t) : nothing
#     end
# end

"""
"""
function Base.show(io::IO, m::MIME"text/plain", diagnostic::Diagnostic)
    print(io, diagnostic.name, ": (spectral=", diagnostic.assumes_spectral_state,
          ", stores_data=", diagnostic.stores_data, ")")
    length(diagnostic.args) != 0 ? print(io, ", args=", diagnostic.args) : nothing
    length(diagnostic.kwargs) != 0 ? print(io, ", kwargs=", diagnostic.kwargs) : nothing
end

# TODO REMOVE OR REWRITE THE THREE METHODS BELOW

# TODO perhaps make more like write_state
function write_data(diagnostic, idx, data, t)
    # TODO better check on ndims
    diagnostic.h5group["data"][fill(:, ndims(data))..., idx] = data
    diagnostic.h5group["t"][idx] = t
end

# TODO perhaps same name as write_local_state, different dispatch
function write_local_data(diagnostic::Diagnostic, idx, data, t)
    if isa(data, AbstractArray)
        diagnostic.data[idx] .= data
    else
        diagnostic.data[idx] = copy(data)
    end
    diagnostic.t[idx] = t
end

"""
"""
function sample_diagnostic!(output, diagnostic, step::Integer, u, prob, t)
    # Check if diagnostic assumes physical field and transform if not yet done
    if !diagnostic.assumes_spectral_state && !output.transformed
        # Transform state
        transform_state!(output, u, get_bwd(prob.domain))
    end

    # Passes the logic onto perform_diagnostic! to do diagnostic and store data
    if diagnostic.assumes_spectral_state
        perform_diagnostic!(diagnostic, step, u, prob, t;
                            store_hdf=output.store_hdf, store_locally=output.store_locally)
    else
        perform_diagnostic!(diagnostic, step, output.U_buffer, prob, t;
                            store_hdf=output.store_hdf, store_locally=output.store_locally)
    end
end

# ---------------------------------- Diagnostics Recipe ------------------------------------

struct DiagnosticRecipe
    name::Symbol
    stride::Int
    storage_limit::String
    kwargs::NamedTuple

    function DiagnosticRecipe(name::Symbol; stride::Int=1, storage_limit="", kwargs...)
        new(name, stride, storage_limit, NamedTuple(kwargs))
    end
end

macro diagnostics(expr)
    if expr.head == :vect  # Vector literal
        return :([$(map(parse_diagnostic_expr, expr.args)...)])
    elseif expr.head != :block
        return :([$(parse_diagnostic_expr(expr))])
    end

    # Multiple expressions in a block
    recipes = Expr[]
    for line in expr.args
        line isa LineNumberNode && continue
        isnothing(line) && continue

        # Handle tuple expressions (when commas are used)
        if line isa Expr && line.head == :tuple
            for item in line.args
                push!(recipes, parse_diagnostic_expr(item))
            end
        else
            push!(recipes, parse_diagnostic_expr(line))
        end
    end

    # Return a vector expression directly
    return Expr(:vect, recipes...)
end

"""

  Allows for single method()
"""
function parse_diagnostic_expr(expr)
    if expr isa Symbol
        return :(DiagnosticRecipe($(QuoteNode(expr))))
    elseif expr isa Expr && expr.head == :call
        name = expr.args[1]

        kwargs = []
        for arg in expr.args[2:end]
            if arg isa Expr && arg.head == :parameters
                # Unwrap the parameters block (from semicolon syntax)
                append!(kwargs, arg.args)
            else
                # Regular keyword argument
                push!(kwargs, arg)
            end
        end

        return :(DiagnosticRecipe($(QuoteNode(name)); $(kwargs...)))
    elseif expr.head == :(=)
        error("Aliases, alias = method(kwargs...), is not supported for diagnostics.")
    else
        error("Invalid diagnostic syntax: $expr")
    end
end

# ---------------------------- Include Implemented Diagnostics -----------------------------

include("CFL.jl")
include("COM.jl")
include("display.jl")
include("energy_integrals.jl")
include("fluxes.jl")
include("probe.jl")
include("profiles.jl")
include("progress.jl")
include("spectral.jl")

# ---------------------------------- Default Diagnostics -----------------------------------

const DEFAULT_DIAGNOSTICS = @diagnostics [progress]

requires_operator(::Val{method}; kwargs...) where {method} = OperatorRecipe[]

function required_operators(diagnostic_recipes::Vector{<:DiagnosticRecipe})
    vcat([requires_operator(Val(recipe.name); recipe.kwargs...)
          for recipe in diagnostic_recipes]...)
end