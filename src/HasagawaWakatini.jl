"""
Includes Domain, Operators, Parameters and Timestepper
"""
# module HasagawaWakatini
# export Domain, Operators, Timestepper
# #include("Helperfunctions.jl")
# #include("Operators.jl")
# #include("Timestepper.jl")
# using .Helperfunctions#: Domain
# using .Operators
# using .Timestepper
# #include("Tmp.jl")
# end # module HasagawaWakatini

using Plots
include("domain.jl")
using .Domains
include("diagnostics.jl")
include("utilities.jl")
using LinearAlgebra
using LaTeXStrings
include("spectralODEProblem.jl")
include("schemes.jl")
include("spectralSolve.jl")