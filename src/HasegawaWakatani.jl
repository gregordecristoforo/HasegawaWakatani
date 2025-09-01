#"""
#Includes Domain, diagnostics, utilities, spectralSolve, spectralODEProblem and schemes
#"""

module HasegawaWakatani

#using DotEnv                     # ✓
#DotEnv.load!()                   # ✓
using Plots                      # ✓
include("domain.jl")             # ✓
using .Domains                   # ✓
# TODO remove hack
export diffX, diffXX, diffY, diffYY, poissonBracket, solvePhi, quadraticTerm,
    diffusion, laplacian, Δ, SpectralOperatorCache, reciprocal, spectral_exp, spectral_expm1,
    spectral_log, hyper_diffusion
using LinearAlgebra              # ✓
using LaTeXStrings               # ✓
include("spectralODEProblem.jl") # ✓
include("schemes.jl")            # ✓   
include("outputer.jl")           # ✓
include("spectralSolve.jl")      # ✓
include("diagnostics.jl")        # ✓
include("utilities.jl")          # ✓

# Main API
export Domain, SpectralODEProblem, spectral_solve, Output
end