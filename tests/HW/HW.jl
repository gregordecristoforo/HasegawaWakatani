## Run all (alt+enter)
include("../../src/HasagawaWakatini.jl")

## Run Hasagawa Wakatani simulations
domain = Domain(128, 128, 2 * pi / 0.15, 2 * pi / 0.15, anti_aliased=true)

#Random initial conditions
ic = initial_condition_linear_stability(domain, 10^-2)

# Linear operator
function L(u, d, p, t)
    D_n = p["D_n"] .* diffusion(u, d) .- p["C"] * u
    D_Ω = p["D_Ω"] .* diffusion(u, d)
    [D_n;;; D_Ω]
end

# Non-linear terms
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dn = -poissonBracket(ϕ, n, d)
    dn .-= p["kappa"] .* diffY(ϕ, d)
    dn .+= p["C"] .* (ϕ)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .+= p["C"] .* (ϕ .- n)
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_n" => 1e-2,
    "D_Ω" => 1e-2,
    "kappa" => 1,
    "C" => 1,
)

t_span = [0, 2000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    #ProbeDensityDiagnostic((0, 0), N=100),
    #PlotDensityDiagnostic(5000),
    RadialFluxDiagnostic(1),
    KineticEnergyDiagnostic(1),
    PotentialEnergyDiagnostic(1),
    GetLogModeDiagnostic(10, :ky),
    CFLDiagnostic(100),
]

# Output
cd("tests/HW")
#output = Output(prob, 1001, diagnostics, "Hasagawa-Wakatani C new scan.h5") 

# Solve and plot
#sol = spectral_solve(prob, MSS3(), output)

## Parameter scan
C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

for C in C_values
    println("Started simulation for C = $(C)!")
    prob.p["C"] = C
    #prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)
    output = Output(prob, 201, diagnostics, "Hasagawa-Wakatani C weekend scan.h5")
    sol = spectral_solve(prob, MSS3(), output)
end