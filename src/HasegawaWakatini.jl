#"""
#Includes Domain, diagnostics, utilities, spectralSolve, spectralODEProblem and schemes
#"""

using DotEnv                     # ✓
DotEnv.load!()                   # ✓
using Plots                      # ✓
include("domain.jl")             # ✓
using .Domains                   # ✓
using LinearAlgebra              # ✓
using LaTeXStrings               # ✓
include("spectralODEProblem.jl") # ✓
include("schemes.jl")            # ✓   
include("outputer.jl")           
include("spectralSolve.jl")      # ✓
include("diagnostics.jl")        
include("utilities.jl")          # ✓