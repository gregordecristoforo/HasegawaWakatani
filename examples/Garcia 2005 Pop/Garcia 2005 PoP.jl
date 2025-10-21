## Run all (alt+enter)
using HasegawaWakatani
using CUDA

domain = Domain(1024, 1024; Lx = 50, Ly = 50, MemoryType = CuArray, precision = Float32)

# Check documentation to see other initial conditions
ic = initial_condition(isolated_blob, domain)

# Linear operator
function Linear(du, u, operators, p, t)
    @unpack κ, ν = p
    θ, Ω = eachslice(u; dims = 3)
    dθ, dΩ = eachslice(du; dims = 3)
    @unpack laplacian = operators
    dθ .= κ .* laplacian(θ)
    dΩ .= ν .* laplacian(Ω)
end

# Non-linear operator
function NonLinear(du, u, operators, p, t)
    θ, Ω = eachslice(u; dims = 3)
    dθ, dΩ = eachslice(du; dims = 3)
    @unpack diff_y, poisson_bracket, solve_phi = operators
    ϕ = solve_phi(Ω)
    dθ .= poisson_bracket(θ, ϕ)
    dΩ .= poisson_bracket(Ω, ϕ) .- diff_y(θ)
end

# Parameters
parameters = (ν = 1e-2, κ = 1e-2)

# Time interval
tspan = [0.0, 20.0]

# The problem
prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan; p = parameters, dt = 1e-3,
                          boussinesq = true, operators = :default,
                          aliases = [:∂x => :diff_x],
                          additional_operators = [OperatorRecipe(:diff_y),
                              OperatorRecipe(:laplacian),
                              OperatorRecipe(:poisson_bracket),
                              OperatorRecipe(:solve_phi)])

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(1),
    #ProgressDiagnostic(100),
    #CFLDiagnostic(1),
    PlotDensityDiagnostic(1000),
    PlotVorticityDiagnostic(1000)
    #PlotPotentialDiagnostic(1000),
]

# Folder path
cd(relpath(@__DIR__, pwd()))

# The output
output = Output(prob; filename = "output/Garcia 2005 PoP.h5", diagnostics = diagnostics,
                stride = -1, simulation_name = :parameters, field_storage_limit = "0.5 GB",
                store_locally = true)

using BenchmarkTools

# Solve and plot
@time sol = spectral_solve(prob, MSS3(), output; resume = false)