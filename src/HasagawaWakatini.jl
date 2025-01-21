#"""
#Includes Domain, diagnostics, utilities, spectralSolve, spectralODEProblem and schemes
#"""

using Plots
include("domain.jl")
using .Domains
include("diagnostics.jl")
include("utilities.jl")
using LinearAlgebra
using LaTeXStrings
include("spectralODEProblem.jl")
include("schemes.jl")
include("outputer.jl")
include("spectralSolve.jl")