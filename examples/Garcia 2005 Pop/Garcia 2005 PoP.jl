## Run all (alt+enter)
using HasegawaWakatani
using CUDA

domain = Domain(1024, 1024, Lx=50, Ly=50, precision=Float32)#, mem=CuArray)

# Check documentation to see other initial conditions
ic = initial_condition(isolated_blob, domain)

# Linear operator
function Linear(du, u, domain, p, t)
    @unpack κ, ν = p
    θ, Ω = eachslice(u, dims=3)
    dθ, dΩ = eachslice(du, dims=3)
    dθ .= κ .* laplacian(θ, domain)
    dΩ .= ν .* laplacian(Ω, domain)
end

# Non-linear operator
function NonLinear(du, u, domain, p, t)
    θ, Ω = eachslice(u, dims=3)
    dθ, dΩ = eachslice(du, dims=3)
    ϕ = solve_phi(Ω, domain)
    dθ .= poisson_bracket(θ, ϕ, domain)
    dΩ .= poisson_bracket(Ω, ϕ, domain) .- diff_y(θ, domain)
end

# Parameters
parameters = (ν=1e-2, κ=1e-2)

# Time interval
tspan = [0.0, 20.0]

# The problem
prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan, p=parameters, dt=1e-3, boussinesq=true)

# Array of diagnostics want
diagnostics = [
    ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(1),
    ProgressDiagnostic(100),
    CFLDiagnostic(1),
    PlotDensityDiagnostic(1000),
    PlotVorticityDiagnostic(1000),
    PlotPotentialDiagnostic(1000),
]

# Folder path
cd(relpath(@__DIR__, pwd()))

# The output
output = Output(prob, filename="output/Garcia 2005 PoP.h5", diagnostics=diagnostics,
    stride=-1, simulation_name=:parameters, field_storage_limit="0.5 GB",
    store_locally=true)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=false)