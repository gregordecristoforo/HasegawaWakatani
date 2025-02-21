## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
#domain = Domain(256, 256, 50, 50, anti_aliased=false)
u0 = gaussian.(domain.x', domain.y, A=0.1, B=1, l=1)

# Linear operator
function L(u, d, p, t)
    D_η = p["kappa"] * diffusion(u, d)
    D_Ω = p["nu"] * diffusion(u, d)
    [D_η;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    η = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dη = -poissonBracket(ϕ, η, d)
    dη += p["kappa"]*quadraticTerm(diffX(η, d), diffX(η, d), d)
    dη += p["kappa"]*quadraticTerm(diffY(η, d), diffY(η, d), d)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ -= diffY(η, d)
    return [dη;;; dΩ]
end

# Parameters
parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

# Time interval
t_span = [0, 20]

# Speed up simulation
FFTW.set_num_threads(16)

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(),
    ProgressDiagnostic(100),
    #CFLDiagnostic(),
    PlotDensityDiagnostic(1000)
]

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3)

# The output
output = Output(prob, 21, diagnostics)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

# Folder path
cd("tests/Garcia 2005 Pop/")

## Recreate max velocity plot
tends = logspace(2, -2, 22)
dts = tends / 2000
amplitudes = logspace(-2, 5, 22)
max_velocities = similar(amplitudes)
velocities = Vector{typeof(data)}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i, A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics)
    # Solve 
    sol = spectral_solve(prob, MSS3(), output)
    # Extract velocity
    velocities[i] = extract_diagnostic(sol.diagnostics[1].data)
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2, :])
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle)
savefig("blob velocities log(n).pdf")