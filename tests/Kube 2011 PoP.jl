## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
#domain = Domain(256, 256, 50, 50, anti_aliased=false)
u0 = log.(gaussian.(domain.x', domain.y, A=1e5, B=1, l=1))

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
    dη += p["kappa"] * quadraticTerm(diffX(η, d), diffX(η, d), d)
    dη += p["kappa"] * quadraticTerm(diffY(η, d), diffY(η, d), d)
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
t_span = [0, 16]

# Speed up simulation
FFTW.set_num_threads(16)

# Inverse transform
function inverse_transformation(u)
    @. u[:,:,1] = exp(u[:,:,1]) - 1
end

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3, 
                            inverse_transformation = inverse_transformation)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(10),
    ProgressDiagnostic(100),
    #CFLDiagnostic(),
    PlotDensityDiagnostic(1000)
]

# The output
output = Output(prob, 21, diagnostics, "Kube 2011 Pop test.h5")

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

plot(output.simulation["RadialCOMDiagnostic/t"][1:200], output.simulation["RadialCOMDiagnostic/data"][2,1:200])

# Folder path
cd("tests/Kube 2011 Pop/")

## Recreate max velocity plot
tends = logspace(2, 0.30, 22)
dts = tends / 10000
amplitudes = logspace(-2, 5, 22)
max_velocities = similar(amplitudes)
velocities = Vector{AbstractArray}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i, A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = log.(gaussian.(domain.x', domain.y, A=A, B=1, l=1))
    # Update problem 
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], [0, tends[i]], p=parameters, dt=dts[i], 
                                inverse_transformation = inverse_transformation)
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(10), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics, "Kube 2011 Pop 1024x1024 max vel.h5")
    # Solve 
    sol = spectral_solve(prob, MSS3(), output)
    # Extract velocity
    velocities[i] = extract_diagnostic(sol.diagnostics[1].data)
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2, :])
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle)
savefig("blob velocities log(n).pdf")

save("max_velocities logn(n).jld", "data", max_velocities)

fid = h5open("Kube 2011 Pop max vel.h5")
maximum(fid["2025-02-23T14:20:21.560"]["fields"][:,:,1,1])
